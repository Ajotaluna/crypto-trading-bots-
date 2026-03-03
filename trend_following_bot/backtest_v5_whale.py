"""
backtest_v5_whale.py — Backtest completo del bot V5 "Trend Rider"

Importa DIRECTAMENTE los módulos de producción:
  - trading_strategy.py  → confirm_entry (8 filtros), PositionManager (PID 4 ATR), calculate_indicators
  - scanner_anomaly.py   → AnomalyScanner (BOOT mode + CONTINUO mode)
  - whale_math_core.py   → whale_score (11 señales, 4 fases)

Configuración bajo prueba (cambios recientes):
  • TRAIL_DISTANCE_ATR   = 4.0    (antes 2.0)
  • MAX_HOLD_CANDLES     = 144    (antes 96)
  • PID Kp=0.15, Kd=0.03         (antes Kp=0.2, Kd=0.1)
  • Kalman q=0.01, r=8.0         (antes q=0.02, r=10.0)
  • confirm_entry: ADX≥22 + kf_slope + MACD hist (todos nuevos)
  • AnomalyScanner: boot_mode en primer scan

Para ejecutar:
  cd trend_following_bot
  python backtest_v5_whale.py

Los CSVs de datos deben estar en:
  trend_following_bot/nascent_scanner/data/   → pares en formato *_15m.csv
  trend_following_bot/nascent_scanner/data_monthly/ → subset para prueba rápida

Estructura mínima de cada CSV:
  timestamp, open, high, low, close, volume, taker_buy_base_asset_volume
  (la columna taker_buy_base_asset_volume activa las señales de CVD en whale_math_core)

Salidas:
  - Imprime día a día en consola
  - Guarda reporte en nascent_scanner/reports/backtest_v5_whale.md
  - Guarda lista de trades en nascent_scanner/reports/backtest_v5_trades.csv
"""

import os
import sys
import time
import numpy as np
import pandas as pd

# ── Añadir el directorio raíz del bot al path para importar módulos de producción
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from trading_strategy import (
    calculate_indicators,
    confirm_entry,
    PositionManager,
    PIDController,
    INITIAL_CAPITAL,
    LEVERAGE,
    COMMISSION,
    RISK_PER_ENTRY,
    MAX_CAPITAL_PER_TRADE,
    MAX_SIGNALS,
    MAX_HOLD_CANDLES,
    DAILY_LOSS_CAP,
    CANDLES_PER_DAY,
    INITIAL_SL_ATR,
    BE_LOCK_ATR,
    TRAIL_DISTANCE_ATR,
    SCALE_LEVEL_2,
    SCALE_LEVEL_3,
    MAX_SCALE,
    MAX_SCORE,
    ENTRY_WINDOW_CANDLES,
)
from scanner_anomaly import AnomalyScanner
from whale_math_core import whale_score
from market_regime import detect_regime, REGIME_RANGING, REGIME_CONFIG

# ──────────────────────────────────────────────────────────────
# CONFIGURACIÓN DEL BACKTEST
# ──────────────────────────────────────────────────────────────

WARMUP_DAYS      = 5      # Días de calentamiento (no se opera, indica la historia mínima)
HISTORY_NEEDED   = 480    # Candles de historia mínima para indicadores estables
TOP_N            = 10     # Picks del AnomalyScanner por día

# Whale scoring mínimo para incluir en watchlist (equivalente a MIN_SCORE del scanner vivo)
# Ajustar según la calidad de los datos disponibles
WHALE_MIN_SCORE  = 130

DATA_FOLDER_DEFAULT = "data"         # Carpeta dentro de nascent_scanner/
REPORT_NAME         = "backtest_v5_whale"
COOLDOWN_CANDLES    = 192            # 48h en 15m — no re-entrar en un par que perdió
BTC_SYMBOL          = "BTCUSDT"      # Par usado como barómetro de régimen


# ──────────────────────────────────────────────────────────────
# CARGADOR DE DATOS
# ──────────────────────────────────────────────────────────────

def load_data(data_dir: str, min_candles: int) -> dict:
    """
    Carga todos los CSVs *_15m.csv del directorio, calcula indicadores
    y filtra los que no tengan suficiente historia.

    Returns: {symbol: DataFrame_con_indicadores}
    """
    pair_data = {}
    if not os.path.exists(data_dir):
        print(f"❌ Directorio no encontrado: {data_dir}")
        print(f"   Crea la carpeta y coloca los CSVs en formato SYMBOL_15m.csv")
        return {}

    files = sorted(f for f in os.listdir(data_dir) if f.endswith("_15m.csv"))
    if not files:
        print(f"❌ No hay archivos *_15m.csv en {data_dir}")
        return {}

    print(f"📂 Cargando {len(files)} pares desde {data_dir}...")
    errors = 0

    for fname in files:
        symbol = fname.replace("_15m.csv", "")
        path   = os.path.join(data_dir, fname)
        try:
            df = pd.read_csv(path)
            df.columns = [c.lower() for c in df.columns]

            # Timestamp
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.sort_values('timestamp', inplace=True)
                df.reset_index(drop=True, inplace=True)

            if len(df) < min_candles:
                continue

            df = calculate_indicators(df)
            pair_data[symbol] = df
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"   ⚠️  {fname}: {e}")

    print(f"✅ {len(pair_data)} pares cargados ({errors} errores)")
    return pair_data


# ──────────────────────────────────────────────────────────────
# WHALE SCORE OFFLINE (sin APIs)
# ──────────────────────────────────────────────────────────────

def whale_score_from_df(df: pd.DataFrame) -> dict:
    """
    Calcula el whale_score a partir solo del DataFrame (sin contexto on-chain).
    En el backtest no hay APIs disponibles, por lo que se usa únicamente
    la señal de klines (CVD, absorción, sweeps, vol z-score, CVD divergence, taker whale).
    """
    try:
        return whale_score(df, context=None)
    except Exception:
        return {'score': 0, 'confidence': 'NONE', 'direction': 'NEUTRAL', 'reasons': []}


# ──────────────────────────────────────────────────────────────
# MOTOR DE SIMULACIÓN
# ──────────────────────────────────────────────────────────────

class BacktestV5:

    def __init__(self, data_dir: str):
        self.data_dir    = data_dir
        self.reports_dir = os.path.join(ROOT, "nascent_scanner", "reports")
        os.makedirs(self.reports_dir, exist_ok=True)

        self.scanner      = AnomalyScanner()
        self.position_mgr = PositionManager()
        self.cooldown     = {}   # {symbol: candle_idx del último stop loss}

    # ── Carga y alineación ──────────────────────────────────────

    def load_aligned_data(self):
        """
        Carga y alinea todos los pares (incluyendo BTC) en un índice cronológico global.
        Retorna (pair_data_dict, timeline_list, btc_df).
        """
        min_len = (WARMUP_DAYS * CANDLES_PER_DAY) + HISTORY_NEEDED
        raw     = load_data(self.data_dir, min_candles=min_len)
        if not raw:
            return {}, [], None

        # ── Cargar BTCUSDT por separado como barómetro (puede estar en la carpeta o no)
        btc_df  = None
        btc_paths = [
            os.path.join(self.data_dir, f"{BTC_SYMBOL}_15m.csv"),
            os.path.join(os.path.dirname(self.data_dir), "data", f"{BTC_SYMBOL}_15m.csv"),
        ]
        for bp in btc_paths:
            if os.path.exists(bp):
                try:
                    btc_df = pd.read_csv(bp)
                    btc_df.columns = [c.lower() for c in btc_df.columns]
                    if 'timestamp' in btc_df.columns:
                        btc_df['timestamp'] = pd.to_datetime(btc_df['timestamp'])
                        btc_df.sort_values('timestamp', inplace=True)
                        btc_df.reset_index(drop=True, inplace=True)
                    from trading_strategy import calculate_indicators
                    btc_df = calculate_indicators(btc_df)
                    print(f"📊 BTC barómetro cargado: {len(btc_df):,} candles desde {bp.split(os.sep)[-1]}")
                except Exception as e:
                    print(f"   ⚠️  BTC error: {e}")
                    btc_df = None
                break

        if btc_df is None:
            print(f"   ⚠️  BTCUSDT no encontrado — régimen será RANGING por defecto (conservador)")

        # ── Construir timeline global
        all_ts = set()
        for df in raw.values():
            if 'timestamp' in df.columns:
                all_ts.update(df['timestamp'].tolist())

        timeline = sorted(all_ts)
        if not timeline:
            timeline = list(range(max(len(df) for df in raw.values())))
            return raw, timeline, btc_df

        full_idx = pd.DatetimeIndex(timeline)
        aligned  = {}
        for sym, df in raw.items():
            if 'timestamp' not in df.columns:
                aligned[sym] = df
                continue
            df_aligned = df.set_index('timestamp').reindex(full_idx)
            df_aligned.reset_index(inplace=True)
            df_aligned.rename(columns={'index': 'timestamp'}, inplace=True)
            aligned[sym] = df_aligned

        return aligned, timeline, btc_df

    # ── Scanner diario ──────────────────────────────────────────

    def _run_daily_scan(
        self,
        pair_data:   dict,
        t_idx:       int,
        is_boot:     bool,
    ) -> list:
        """
        Ejecuta el scanner anomaly (con modo BOOT en el primer día)
        y el whale scorer offline.
        Retorna la watchlist del día combinada y deduplicada.
        """
        # 1. AnomalyScanner
        eligible = {}
        for sym, df in pair_data.items():
            slice_df = df.iloc[:t_idx].dropna(subset=['close'])
            if len(slice_df) >= 96:
                eligible[sym] = slice_df

        anomaly_picks = self.scanner.score_universe(
            eligible, now_idx=-1, top_n=TOP_N, boot_mode=is_boot
        )

        # 2. Whale score offline — SOLO sobre candidatos del anomaly scanner
        #    Antes corría sobre todos los pares O(n=500) → demasiado lento
        anomaly_syms = {p['symbol'] for p in anomaly_picks}

        # Complementar con top-20 por volumen (pares no capturados por anomaly)
        complementary = sorted(
            [(s, float(d['volume'].iloc[-1])) for s, d in eligible.items()
             if s not in anomaly_syms and len(d) > 0],
            key=lambda x: x[1], reverse=True
        )[:20]

        whale_candidates = list(anomaly_syms) + [s for s, _ in complementary]

        whale_picks = []
        for sym in whale_candidates:
            if sym not in eligible:
                continue
            try:
                ws = whale_score_from_df(eligible[sym].tail(200))
            except Exception:
                continue
            if ws['score'] >= WHALE_MIN_SCORE and ws['confidence'] in ('HIGH', 'ULTRA'):
                whale_picks.append({
                    'symbol':     sym,
                    'score':      ws['score'],
                    'direction':  ws['direction'],
                    'reasons':    ', '.join(ws['reasons'][:3]),
                    'confidence': ws['confidence'],
                    'layer':      'WHALE',
                })

        # 3. Merge sin duplicados (por orden de score)
        all_picks  = anomaly_picks + whale_picks
        all_picks.sort(key=lambda x: x['score'], reverse=True)

        seen    = set()
        watchlist = []
        for p in all_picks:
            if p['symbol'] not in seen and p['direction'] != 'NEUTRAL':
                watchlist.append(p)
                seen.add(p['symbol'])

        return watchlist

    # ── Loop principal ──────────────────────────────────────────

    def run(self):
        print("\n" + "=" * 70)
        print("  BACKTEST V5 WHALE — Trend Rider + Market Regime Filter")
        print(f"  Trail: {TRAIL_DISTANCE_ATR} ATR | Hold: {MAX_HOLD_CANDLES} candles ({MAX_HOLD_CANDLES*15/60:.0f}h)")
        print(f"  SL: {INITIAL_SL_ATR} ATR | BE Lock: {BE_LOCK_ATR} ATR | Cooldown: {COOLDOWN_CANDLES*15//60}h")
        print(f"  confirm_entry: ADX≥22 + kf_slope + MACD (8 filtros) + Régimen BTC")
        print(f"  Capital: ${INITIAL_CAPITAL:,} | Leverage: {LEVERAGE}x | Risk: {RISK_PER_ENTRY*100:.1f}%/entry")
        print("=" * 70)

        t0 = time.time()
        pair_data, timeline, btc_df = self.load_aligned_data()
        if not pair_data or not timeline:
            print("❌ Sin datos. Abortando.")
            return

        GLOBAL_WARMUP = WARMUP_DAYS * CANDLES_PER_DAY + HISTORY_NEEDED
        test_start    = GLOBAL_WARMUP
        n_candles     = len(timeline)
        test_days     = (n_candles - test_start) // CANDLES_PER_DAY

        print(f"\n📅 Período: {n_candles} candles | Test: {test_days} días")
        print(f"   Pares: {len(pair_data)} | Warmup: {GLOBAL_WARMUP} candles")
        print()

        equity      = float(INITIAL_CAPITAL)
        all_trades  = []
        daily_logs  = []

        is_boot           = True
        current_watchlist = []
        current_regime    = {'regime': REGIME_RANGING, 'config': REGIME_CONFIG[REGIME_RANGING],
                             'description': 'Init', 'adx': 0, 'kf_slope': 0, 'atr_ratio': 1}

        for day_idx in range(test_days):
            t_idx        = test_start + day_idx * CANDLES_PER_DAY
            day_eq_start = equity

            try:
                ts_label = str(timeline[t_idx])[:10]
            except Exception:
                ts_label = f"Day {day_idx + 1}"

            # ── 1. Detectar régimen de mercado usando BTC ───────────────
            if btc_df is not None:
                # Encontrar el índice en BTC alineado con el candle actual
                btc_slice_idx = min(t_idx, len(btc_df) - 1)
                current_regime = detect_regime(btc_df, candle_idx=btc_slice_idx)
            regime_cfg  = current_regime['config']
            max_slots   = regime_cfg['max_signals']
            long_bias   = regime_cfg['long_bias']
            short_bias  = regime_cfg['short_bias']
            size_mult   = regime_cfg['size_mult']
            regime_name = current_regime['regime']

            # ── 2. Scan diario (anomaly + whale) ───────────────────
            raw_watchlist = self._run_daily_scan(pair_data, t_idx, is_boot=is_boot)
            is_boot = False

            # Aplicar bias del régimen al score de cada pick
            current_watchlist = []
            for pick in raw_watchlist:
                p = dict(pick)
                if p['direction'] == 'LONG':
                    p['score'] = int(p['score'] * long_bias)
                elif p['direction'] == 'SHORT':
                    p['score'] = int(p['score'] * short_bias)
                if p['score'] > 0:
                    current_watchlist.append(p)
            current_watchlist.sort(key=lambda x: x['score'], reverse=True)

            # ── 3. Ciclo candle a candle ─────────────────────
            day_longs   = 0
            day_shorts  = 0
            day_stopped = False
            attempted_today = set()

            for offset in range(CANDLES_PER_DAY):
                candle_idx = t_idx + offset
                if candle_idx >= n_candles:
                    break

                # Daily loss cap
                if equity < day_eq_start * (1 - DAILY_LOSS_CAP):
                    day_stopped = True
                    break

                # 1. Gestionar posiciones abiertas
                closed = self.position_mgr.update_positions(pair_data, candle_idx)
                for trade in closed:
                    trade['exit_time'] = timeline[candle_idx] if candle_idx < len(timeline) else None
                    equity += trade['pnl_dollar']
                    all_trades.append(trade)
                    # ── Registrar cooldown si fue un stop loss sin BE ────────────
                    if trade['reason'] == 'STOP_LOSS':
                        self.cooldown[trade['symbol']] = candle_idx

                # 2. Scaling en posiciones existentes
                for sym in list(self.position_mgr.positions.keys()):
                    if sym not in pair_data:
                        continue
                    df = pair_data[sym]
                    if candle_idx >= len(df):
                        continue
                    candle = df.iloc[candle_idx]
                    if pd.isna(candle['close']) or candle['close'] == 0:
                        continue
                    price   = float(candle['close'])
                    atr_val = float(candle.get('atr', 0))
                    if atr_val <= 0:
                        continue
                    should_scale, _ = self.position_mgr.check_scale_opportunity(sym, price, atr_val)
                    if should_scale:
                        pos      = self.position_mgr.positions[sym]
                        risk_amt = equity * RISK_PER_ENTRY
                        risk_d   = abs(price - pos['sl'])
                        if risk_d > 0:
                            v = (risk_amt / (risk_d / price)) / LEVERAGE
                            max_add = max(0.0, equity * MAX_CAPITAL_PER_TRADE - pos['total_amount'])
                            v = min(v, max_add)
                            if v >= 5:
                                self.position_mgr.add_to_position(sym, price, v, candle_idx)

                # 3. Nuevas entradas (solo en la ventana inicial del día)
                if offset >= ENTRY_WINDOW_CANDLES:
                    continue
                if len(self.position_mgr.positions) >= max_slots:   # usa slots del régimen
                    continue

                for pick in current_watchlist:
                    sym = pick['symbol']
                    if sym in self.position_mgr.positions:
                        continue
                    if sym in attempted_today:
                        continue
                    if len(self.position_mgr.positions) >= max_slots:
                        break
                    if sym not in pair_data:
                        continue

                    # ── Cooldown: ignorar par si tuvo stop reciente ────────────
                    last_stop = self.cooldown.get(sym, -9999)
                    if candle_idx - last_stop < COOLDOWN_CANDLES:
                        continue

                    df = pair_data[sym]
                    if candle_idx >= len(df):
                        continue

                    df_slice = df.iloc[:candle_idx + 1].dropna(subset=['close'])
                    if len(df_slice) < 30:    # confirm_entry necesita ≥ 30 candles
                        continue

                    direction = pick.get('direction', 'LONG')
                    if direction == 'NEUTRAL':
                        continue

                    if not confirm_entry(df_slice, direction):
                        continue

                    # Ejecutar entrada
                    attempted_today.add(sym)
                    entry_price = float(df_slice['close'].iloc[-1])
                    atr_val     = float(df_slice['atr'].iloc[-1])
                    if atr_val <= 0:
                        continue

                    sl = (
                        entry_price - atr_val * INITIAL_SL_ATR if direction == 'LONG'
                        else entry_price + atr_val * INITIAL_SL_ATR
                    )

                    risk_d   = abs(entry_price - sl)
                    if risk_d == 0:
                        continue

                    # Aplicar size_mult del régimen
                    risk_amt  = equity * RISK_PER_ENTRY * size_mult
                    pos_value = min(
                        (risk_amt / (risk_d / entry_price)) / LEVERAGE,
                        equity * MAX_CAPITAL_PER_TRADE,
                    )
                    if pos_value < 5:
                        continue

                    self.position_mgr.open_position(
                        sym, direction, entry_price, sl,
                        pos_value, atr_val, candle_idx
                    )
                    if direction == 'LONG':
                        day_longs += 1
                    else:
                        day_shorts += 1

            # ── Resumen del día ───────────────────────────────────
            day_trades = [t for t in all_trades
                          if t.get('exit_time') is not None
                          and str(t['exit_time'])[:10] == ts_label]
            day_pnl    = equity - day_eq_start
            day_wins   = sum(1 for t in day_trades if t['pnl_dollar'] > 0)
            day_losses = sum(1 for t in day_trades if t['pnl_dollar'] <= 0)

            daily_logs.append({
                'day':     day_idx + 1,
                'date':    ts_label,
                'longs':   day_longs,
                'shorts':  day_shorts,
                'wins':    day_wins,
                'losses':  day_losses,
                'pnl':     day_pnl,
                'equity':  equity,
                'open':    len(self.position_mgr.positions),
                'stopped': day_stopped,
                'regime':  regime_name,
                'adx':     current_regime.get('adx', 0),
            })

            emoji   = "📈" if day_pnl > 0 else ("📉" if day_pnl < 0 else "➡️")
            stop_s  = " ⛔STOP" if day_stopped else ""
            open_s  = f" [Open:{len(self.position_mgr.positions)}]" if self.position_mgr.positions else ""
            rg_icon = {"BULL": "🟢", "BEAR": "🔴", "RANGING": "🔵", "VOLATILE": "🟡"}.get(regime_name, "⚪")
            print(
                f"  {emoji} Day {day_idx+1:3d} | {ts_label} | {rg_icon}{regime_name:8s} ADX:{current_regime.get('adx',0):4.1f} | "
                f"New:{day_longs}L/{day_shorts}S | "
                f"Closed:{day_wins}W/{day_losses}L | "
                f"P&L:{day_pnl:+7.2f} | Equity:{equity:,.2f}"
                f"{open_s}{stop_s}"
            )

        # ── Cerrar posiciones abiertas al final ───────────────────
        for sym in list(self.position_mgr.positions.keys()):
            if sym in pair_data:
                df   = pair_data[sym]
                last = min(test_start + test_days * CANDLES_PER_DAY - 1, len(df) - 1)
                ep   = float(df.iloc[last]['close'])
                t    = self.position_mgr._close_position(sym, ep, 'END_OF_TEST', last)
                equity += t['pnl_dollar']
                all_trades.append(t)

        elapsed = time.time() - t0
        print(f"\n⏱️ Backtest completado en {elapsed:.1f}s")

        self._report(equity, all_trades, daily_logs)

    # ── Reporte ─────────────────────────────────────────────────

    def _report(self, final_equity: float, trades: list, daily_logs: list):
        if not trades:
            print("\n⚠️  Sin trades ejecutados. Verifica los datos y los umbrales.")
            return

        roi        = (final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        wins       = [t for t in trades if t['pnl_dollar'] > 0]
        losses     = [t for t in trades if t['pnl_dollar'] <= 0]
        win_rate   = len(wins) / len(trades) * 100 if trades else 0
        avg_win    = np.mean([t['pnl_dollar'] for t in wins])    if wins    else 0
        avg_loss   = np.mean([t['pnl_dollar'] for t in losses])  if losses  else 0
        rr         = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        avg_hold   = np.mean([t['hold_candles'] for t in trades]) * 15 / 60
        med_hold   = np.median([t['hold_candles'] for t in trades]) * 15 / 60
        t_per_day  = len(trades) / len(daily_logs) if daily_logs else 0

        peak, max_dd = INITIAL_CAPITAL, 0.0
        for d in daily_logs:
            peak  = max(peak, d['equity'])
            dd    = (d['equity'] - peak) / peak * 100
            max_dd = min(max_dd, dd)

        win_days   = sum(1 for d in daily_logs if d['pnl'] > 0)
        long_ts    = [t for t in trades if t['direction'] == 'LONG']
        short_ts   = [t for t in trades if t['direction'] == 'SHORT']
        long_wr    = sum(1 for t in long_ts if t['pnl_dollar'] > 0) / len(long_ts) * 100 if long_ts else 0
        short_wr   = sum(1 for t in short_ts if t['pnl_dollar'] > 0) / len(short_ts) * 100 if short_ts else 0

        best  = max(trades, key=lambda t: t['pnl_dollar'])
        worst = min(trades, key=lambda t: t['pnl_dollar'])

        # Exit reasons
        reasons = {}
        for t in trades:
            r = t['reason']
            if r not in reasons:
                reasons[r] = {'n': 0, 'pnl': 0.0, 'wins': 0}
            reasons[r]['n']    += 1
            reasons[r]['pnl']  += t['pnl_dollar']
            if t['pnl_dollar'] > 0:
                reasons[r]['wins'] += 1

        # ── Imprimir resumen ──────────────────────────────────────
        print("\n" + "=" * 70)
        print("  RESULTADOS FINALES — BACKTEST V5 WHALE")
        print("=" * 70)
        print(f"  Capital Inicial : ${INITIAL_CAPITAL:>10,.2f}")
        print(f"  Capital Final   : ${final_equity:>10,.2f}  ({roi:+.1f}%)")
        print(f"  Max Drawdown    : {max_dd:.1f}%")
        print(f"  Días Ganadores  : {win_days}/{len(daily_logs)}")
        print()
        print(f"  Total Trades    : {len(trades)} ({t_per_day:.1f}/día)")
        print(f"  Win Rate        : {win_rate:.1f}%  ({len(wins)}/{len(trades)})")
        print(f"  Avg Win         : ${avg_win:+.2f}")
        print(f"  Avg Loss        : ${avg_loss:+.2f}")
        print(f"  Risk/Reward     : {rr:.2f}R")
        print(f"  Hold Avg/Med    : {avg_hold:.1f}h / {med_hold:.1f}h")
        print()
        print(f"  LONG  : {len(long_ts)} trades | WR: {long_wr:.0f}%")
        print(f"  SHORT : {len(short_ts)} trades | WR: {short_wr:.0f}%")
        print()
        print(f"  Mejor trade  : {best['symbol']} {best['direction']}  ${best['pnl_dollar']:+,.2f}")
        print(f"  Peor trade   : {worst['symbol']} {worst['direction']}  ${worst['pnl_dollar']:+,.2f}")
        print()
        print("  RAZONES DE SALIDA:")
        for r, d in sorted(reasons.items(), key=lambda x: x[1]['pnl'], reverse=True):
            wr_r = d['wins'] / d['n'] * 100 if d['n'] else 0
            print(f"    {r:20s} : {d['n']:4d} trades | WR:{wr_r:.0f}% | P&L:${d['pnl']:+,.2f}")

        # ── Construir reporte Markdown ────────────────────────────
        lines = [
            f"# Backtest V5 Whale — {len(daily_logs)} días",
            f"",
            f"## Configuración",
            f"| Parámetro | Valor |",
            f"|-----------|-------|",
            f"| Capital inicial | ${INITIAL_CAPITAL:,} |",
            f"| Leverage | {LEVERAGE}x |",
            f"| Risk/entry | {RISK_PER_ENTRY*100:.1f}% |",
            f"| Trail ATR | {TRAIL_DISTANCE_ATR} |",
            f"| Max Hold | {MAX_HOLD_CANDLES} velas ({MAX_HOLD_CANDLES*15/60:.0f}h) |",
            f"| SL inicial | {INITIAL_SL_ATR} ATR |",
            f"| BE Lock | {BE_LOCK_ATR} ATR |",
            f"| confirm_entry | ADX≥22 + kf_slope + MACD + 5 más |",
            f"| Whale MIN_SCORE | {WHALE_MIN_SCORE} |",
            f"",
            f"## Resumen",
            f"| Métrica | Valor |",
            f"|---------|-------|",
            f"| Capital final | ${final_equity:,.2f} |",
            f"| ROI | {roi:+.1f}% |",
            f"| Max Drawdown | {max_dd:.1f}% |",
            f"| Días ganadores | {win_days}/{len(daily_logs)} |",
            f"| Total trades | {len(trades)} ({t_per_day:.1f}/día) |",
            f"| Win Rate | {win_rate:.1f}% |",
            f"| Avg Win | ${avg_win:+.2f} |",
            f"| Avg Loss | ${avg_loss:+.2f} |",
            f"| Risk/Reward | {rr:.2f}R |",
            f"| Hold promedio | {avg_hold:.1f}h |",
            f"| Hold mediana | {med_hold:.1f}h |",
            f"| LONG WR | {long_wr:.0f}% ({len(long_ts)} trades) |",
            f"| SHORT WR | {short_wr:.0f}% ({len(short_ts)} trades) |",
            f"| Mejor trade | {best['symbol']} {best['direction']} ${best['pnl_dollar']:+,.2f} |",
            f"| Peor trade | {worst['symbol']} {worst['direction']} ${worst['pnl_dollar']:+,.2f} |",
            f"",
            f"## Razones de salida",
            f"| Razón | Trades | WR | P&L |",
            f"|-------|--------|----|-----|",
        ]
        for r, d in sorted(reasons.items(), key=lambda x: x[1]['pnl'], reverse=True):
            wr_r = d['wins'] / d['n'] * 100 if d['n'] else 0
            lines.append(f"| {r} | {d['n']} | {wr_r:.0f}% | ${d['pnl']:+,.2f} |")

        lines += [
            f"",
            f"## Régimen de mercado por día",
            f"| Día | Fecha | Régimen | ADX | New L/S | Closed W/L | P&L | Equity |",
            f"|-----|-------|---------|-----|---------|------------|-----|--------|",
        ]
        rg_icons = {"BULL": "🟢", "BEAR": "🔴", "RANGING": "🔵", "VOLATILE": "🟡"}
        for d in daily_logs:
            emoji = "+" if d['pnl'] > 0 else ("-" if d['pnl'] < 0 else "=")
            stop  = " ⛔" if d['stopped'] else ""
            rgi   = rg_icons.get(d.get('regime', ''), '⚪')
            lines.append(
                f"| {emoji}{d['day']:3d} | {d['date']} | {rgi}{d.get('regime','?'):8s} | {d.get('adx',0):4.1f} | "
                f"{d['longs']}L/{d['shorts']}S | "
                f"{d['wins']}W/{d['losses']}L | "
                f"${d['pnl']:+,.2f} | ${d['equity']:,.2f}{stop} |"
            )

        lines += [
            f"",
            f"## Veredicto",
            f"> ${INITIAL_CAPITAL:,} → ${final_equity:,.2f} ({roi:+.1f}%) en {len(daily_logs)} días.",
            f"> {len(trades)} trades ({t_per_day:.1f}/día) | WR: {win_rate:.1f}% | RR: {rr:.2f}R | MaxDD: {max_dd:.1f}%",
        ]

        # ── Guardar reporte Markdown ───────────────────────────────
        md_path = os.path.join(self.reports_dir, f"{REPORT_NAME}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"\n📄 Reporte guardado: {md_path}")

        # ── Guardar CSV de trades ───────────────────────────────────
        if trades:
            trades_df = pd.DataFrame(trades)
            csv_path  = os.path.join(self.reports_dir, f"{REPORT_NAME}_trades.csv")
            trades_df.to_csv(csv_path, index=False)
            print(f"📊 Trades CSV: {csv_path}")


# ──────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n📂 Selecciona fuente de datos:")
    print("  1. data          — Historia completa (lento)")
    print("  2. data_monthly  — Mes más reciente (rápido, ideal para primera prueba)")
    choice = input("  Opción [1/2] (default 1): ").strip()

    folder = "data_monthly" if choice == "2" else DATA_FOLDER_DEFAULT
    data_dir = os.path.join(ROOT, "nascent_scanner", folder)

    print(f"\n  → Usando: {folder}\n")
    BacktestV5(data_dir).run()
