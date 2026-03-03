"""
Main Bot Logic - REFACTORED FOR DOUBLE FUNNEL STRATEGY
Integrates AnomalyScanner (Macro) and TradingStrategy (Micro).
ALIGNED WITH backtest_progressive.py PositionManager logic.
"""
import asyncio
import logging
import sys
import os
import time
from datetime import datetime, timedelta

# Local imports
from config import config
from market_data import MarketData
from win_rate_tracker import WinRateTracker
import pandas as pd
from blacklist_manager import BlacklistManager

# Strategy imports (shared with backtest)
from scanner_anomaly import AnomalyScanner
from whale_market_scanner import scan_whale_universe
from whale_watcher import WhaleWatcher
from orderbook_streamer import OrderbookStreamer
from market_regime import get_live_regime, REGIME_RANGING, REGIME_CONFIG
from trading_strategy import (
    confirm_entry, 
    calculate_indicators, 
    PositionManager,
    PIDController,
    MAX_SCORE,
    TOP_N,
    INITIAL_CAPITAL,
    LEVERAGE,
    COMMISSION,
    RISK_PER_ENTRY,
    MAX_CAPITAL_PER_TRADE,
    MAX_SIGNALS,
    MAX_HOLD_CANDLES,
    DAILY_LOSS_CAP,
    INITIAL_SL_ATR,
    BE_LOCK_ATR,
    TRAIL_DISTANCE_ATR,
    SCALE_LEVEL_2,
    SCALE_LEVEL_3,
    MAX_SCALE
)

# Slots independientes por scanner: cada canal tiene su propio cupo
# Total max posiciones abiertas = ANOMALY_SLOTS + WHALE_SLOTS = 6
ANOMALY_SLOTS = 3   # Max posiciones desde AnomalyScanner
WHALE_SLOTS   = 3   # Max posiciones desde Whale Scanner (HIGH + ULTRA)

# Setup logging
if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/trend_bot.log")
    ]
)
logger = logging.getLogger("TrendBot")

import concurrent.futures

class TrendBot:
    def __init__(self, is_dry_run=True, api_key=None, api_secret=None):
        self.market = MarketData(is_dry_run, api_key, api_secret)
        self.blacklist = BlacklistManager()
        self.running = True
        self.start_balance = 0.0
        
        # Strategies
        self.scanner = AnomalyScanner()

        # Boot mode flag — primer scan ignora tendencias ya activas
        self.is_first_scan = True
        self.current_regime = {'regime': REGIME_RANGING, 'config': REGIME_CONFIG[REGIME_RANGING],
                               'description': 'Init', 'adx': 0}

        # Whale scanner
        self.whale_watchlist: list = []   # Top-15 pares con señal de ballena
        self.whale_watcher = WhaleWatcher()
        
        self.ob_streamer = OrderbookStreamer(depth=20, update_speed="100ms")

        self.daily_watchlist = []  # Top-10 anomaly + hasta 15 whale (sin duplicados)
        self.pos_state = {}  # Local state tracker
        # Tracking de layer por posicion abierta: {symbol: 'ANOMALY'|'WHALE'}
        self.pos_layer: dict = {}
        
        # SCOREBOARD
        self.tracker = WinRateTracker()
        self.tracker.log_summary()

        # ThreadPool & Locks
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    async def start(self):
        """Main Entry Point & Orchestrator"""
        mode_str = "PRODUCTION (REAL MONEY)" if not self.market.is_dry_run else "DRY RUN (PAPER TRADING)"
        logger.info(f"\n{'='*50}\n>>> STARTING TREND BOT V5 (DOUBLE FUNNEL): {mode_str} <<<\n{'='*50}")
        
        # Initialize the global entry lock here to attach it to the running event loop
        self.entry_lock = asyncio.Lock()
        
        # 1. Initialize Balance
        if not self.market.is_dry_run:
             status = await self.market.get_real_account_status()
             if status:
                 self.start_balance = status['equity']
                 logger.info(f"Initial Equity: ${self.start_balance:.2f}")
        else:
             self.start_balance = self.market.balance
        
        self.current_day_str = datetime.utcnow().strftime('%Y-%m-%d')
             
        # LOGGING CAPITAL
        daily_target = self.start_balance * 0.03
        logger.info(f"💰 BALANCE: ${self.start_balance:.2f} | 🎯 DAILY TARGET (3%): ${daily_target:.2f}")

        # 2. Launch Background Loops
        asyncio.create_task(self.ob_streamer.run_stream())
        asyncio.create_task(self.reporting_loop())
        asyncio.create_task(self.safety_loop())
        asyncio.create_task(
            self.whale_watcher.start(self.whale_watchlist, self.market)
        )
        asyncio.create_task(self.whale_entry_loop())

        # 3. Start The Core Brain
        await self.execution_loop()

    async def execution_loop(self):
        """
        The Double Funnel Engine:
        1. Macro Scan (Once per day 00:00 UTC) -> Selects Top 10.
        2. Micro Scan (Continuous/15m) -> Checks confirmation on Top 10.
        """
        logger.info("🧠 Started Execution Loop (Double Funnel)...")
        
        # Initial Macro Scan
        await self.run_macro_scan()
        
        last_scan_date = datetime.utcnow().strftime('%Y-%m-%d')
        last_scan_time = datetime.utcnow()
        MACRO_SCAN_INTERVAL_HOURS = 4
        
        while self.running:
            try:
                now_utc = datetime.utcnow()
                now_date = now_utc.strftime('%Y-%m-%d')
                
                # --- A. MACRO RE-SCAN (Every 4 Hours OR New Day) ---
                hours_since_last_scan = (now_utc - last_scan_time).total_seconds() / 3600
                mins_since_last_scan = (now_utc - last_scan_time).total_seconds() / 60
                is_completely_idle = len(self.market.positions) == 0
                
                if now_date != last_scan_date or hours_since_last_scan >= MACRO_SCAN_INTERVAL_HOURS:
                    if now_date != last_scan_date:
                        logger.info(f"📅 NEW DAY {now_date}: Running Macro Scan...")
                        # Reset Daily PnL
                        self.start_balance = await self.get_equity()
                        self.tracker.reset_daily_pnl_if_new_day()
                    else:
                        logger.info(f"⏱️ INTRADAY SCAN: {hours_since_last_scan:.1f} hours passed. Refreshing market data...")
                        
                    await self.run_macro_scan()
                    last_scan_date = now_date
                    last_scan_time = now_utc

                # --- A.5 IDLE RESCAN (Operational Silence) ---
                elif is_completely_idle and mins_since_last_scan >= 45.0:
                    logger.info(f"🦇 SILENCIO OPERACIONAL: Sin operaciones abiertas por {mins_since_last_scan:.1f} min. Forzando Escaneo Táctico...")
                    await self.run_macro_scan()
                    last_scan_time = now_utc

                # --- B. MICRO SCAN (Intraday) ---
                if not self.daily_watchlist:
                    logger.debug("Silencio: Watchlist vacía. Esperando al próximo ciclo táctico...")
                    await asyncio.sleep(60)
                    continue

                logger.info(f"🔍 MICRO SCAN: Checking {len(self.daily_watchlist)} watchlist picks | Open: {len(self.market.positions)}/{MAX_SIGNALS}")
                
                # Check all picks concurrently
                async def process_pick(pick):
                    symbol = pick['symbol']
                    direction = pick['direction']

                    # Pares whale esperan su movimiento — NO entran por el micro scan normal
                    if pick.get('layer') == 'WHALE':
                        return None

                    # Capacity Check
                    if len(self.market.positions) >= MAX_SIGNALS:
                        return None

                    # Skip if already open
                    if symbol in self.market.positions:
                        return None

                    # MICRO CHECK (solo pares ANOMALY)
                    async with self.entry_lock:
                        # Double-check capacity inside lock
                        if len(self.market.positions) >= MAX_SIGNALS:
                            return None
                        await self.check_micro_entry(symbol, direction, pick)
                    
                    # Return pick if entered to remove it from watchlist
                    if symbol in self.market.positions:
                        return pick
                    return None

                tasks = [process_pick(p) for p in list(self.daily_watchlist)]
                results = await asyncio.gather(*tasks)

                # Remove entered picks from watchlist
                for pick in results:
                    if pick and pick in self.daily_watchlist:
                        self.daily_watchlist.remove(pick)

                await asyncio.sleep(60)

            except Exception as e:
                logger.error(f"Execution Loop Error: {e}")
                await asyncio.sleep(60)

    async def run_macro_scan(self):
        """Double Funnel Macro Scan: Anomaly Scanner (nacientes) + Whale Scanner."""
        try:
            self.daily_watchlist = []
            boot = self.is_first_scan
            scan_mode = "BOOT (ignorar tendencias >4h)" if boot else "CONTINUO (detectar nacientes ≤3 velas)"
            logger.info(f"🔭 ANOMALY SCAN [{scan_mode}]...")

            # ─── RÉGIMEN DE MERCADO: barómetro BTC antes de cualquier scan ─
            regime = await get_live_regime(self.market)
            self.current_regime = regime
            regime_cfg  = regime['config']
            max_slots   = regime_cfg['max_signals']
            long_bias   = regime_cfg['long_bias']
            short_bias  = regime_cfg['short_bias']
            size_mult   = regime_cfg['size_mult']
            logger.info(
                f"📊 RÉGIMEN: {regime['regime']} | ADX:{regime['adx']} | "
                f"ATR×{regime['atr_ratio']} | {regime['description']}"
            )
            logger.info(f"   Slots: {max_slots} | LongBias: {long_bias}x | ShortBias: {short_bias}x | Size: {size_mult}x")

            # ─── ANOMALY SCAN: buscar nacientes (usa klines del mercado) ─
            try:
                # Obtener klines de los pares del universo para el scanner
                universe = await self.market.get_trading_universe()
                if not universe:
                    logger.warning("Anomaly Scan: get_trading_universe() retorno None o vacio — saltando")
                    anomaly_picks = []
                else:
                    universe = universe[:100]  # Analizar top-100 por volumen como muestra
                    pair_data = {}
                    sem = asyncio.Semaphore(8)

                    async def _fetch_klines(sym):
                        try:
                            async with sem:
                                df = await self.market.get_klines(sym, interval='15m', limit=200)
                                if df is not None and not df.empty and len(df) >= 96:
                                    pair_data[sym] = df
                        except Exception:
                            pass  # Error en un par individual no cancela el resto

                    await asyncio.gather(*[_fetch_klines(s) for s in universe])
                    logger.info(f"📡 Anomaly Scan: {len(pair_data)}/{len(universe)} pares con datos listos")

                    anomaly_picks = await asyncio.to_thread(
                        self.scanner.score_universe,
                        pair_data,
                        200,   # now_idx
                        10,    # top_n
                        None,  # long_ratio
                        boot,  # boot_mode
                    )
                    # Asignar confidence si no viene del scorer (modo boot)
                    for p in anomaly_picks:
                        p.setdefault('confidence', 'HIGH')
                    logger.info(f"📡 Anomaly picks: {len(anomaly_picks)} ({'boot' if boot else 'live'} mode)")
                    for p in anomaly_picks[:5]:
                        logger.info(f"  [ANOMALY] {p['symbol']:12s} ({p['direction']:5s}) score={p['score']} conf={p.get('confidence','?')} | {p.get('reasons','')[:60]}")
            except Exception as ae:
                logger.error(f"Anomaly Scan Error: {ae}", exc_info=True)
                anomaly_picks = []

            # Marcar que el primer scan ya ocurrió
            if self.is_first_scan:
                self.is_first_scan = False

            # Añadir anomaly picks al watchlist primero
            self.daily_watchlist = list(anomaly_picks)

            # ─── WHALE SCAN: top-15 adicionales ──────────────────────
            logger.info("🐋 WHALE SCAN: escaneando todo el mercado en batches de 50...")
            try:
                whale_picks = await scan_whale_universe(
                    self.market, top_n=15, ob_streamer=self.ob_streamer
                )
            except Exception as we:
                logger.error(f"Whale Scan Error: {we}")
                whale_picks = []

            # MEDIUM, HIGH y ULTRA ejecutan trades. Solo LOW y NONE se descartan.
            whale_picks_executable = [
                p for p in whale_picks
                if p.get('confidence') in ('MEDIUM', 'HIGH', 'ULTRA')
            ]
            whale_picks_low = [
                p for p in whale_picks
                if p.get('confidence') not in ('MEDIUM', 'HIGH', 'ULTRA')
            ]
            logger.info(
                f"🐋 Whale picks totales: {len(whale_picks)} | "
                f"MEDIUM+HIGH+ULTRA (ejecutan): {len(whale_picks_executable)} | "
                f"LOW/NONE (descartados): {len(whale_picks_low)}"
            )
            # Watcher monitorea todos pero solo MEDIUM+ entran al watchlist
            self.whale_watchlist = whale_picks_executable
            self.whale_watcher.update_pairs(whale_picks_executable)

            # ── WATCHLIST FINAL: slots garantizados por scanner ──────────────
            # Cada scanner tiene su propia sección - NO compiten por espacio.
            # Anomaly: hasta 10 picks  |  Whale: hasta 15 picks
            # Total posible: 25 en watchlist, máx 6 posiciones abiertas.
            watchlist_anomaly = anomaly_picks[:10]   # ya tienen layer='ANOMALY'
            watchlist_whale   = []
            anomaly_syms = {p['symbol'] for p in watchlist_anomaly}
            for wp in whale_picks_executable:
                if wp['symbol'] not in anomaly_syms:  # evitar duplicado si el mismo par salió en ambos
                    watchlist_whale.append(wp)
                if len(watchlist_whale) >= 15:
                    break

            # Construir watchlist final: anomaly primero, luego whale
            self.daily_watchlist = watchlist_anomaly + watchlist_whale
            self.daily_watchlist.sort(key=lambda x: x['score'], reverse=True)

            added_anomaly = len(watchlist_anomaly)
            added_whale   = len(watchlist_whale)


            # --- 📡 ORDERBOOK STREAMER: SUSCRIBIR AL TOP WATCHLIST + ACTIVAS ---
            top_symbols = [p['symbol'] for p in self.daily_watchlist]
            active_symbols = list(self.market.positions.keys())
            symbols_to_track = list(set(top_symbols + active_symbols))
            await self.ob_streamer.update_focus_list(symbols_to_track)
            
            # WARM-UP: Dale 3 segundos al OrderBook Streamer para descargar los mapas de liquidez de Binance
            logger.info("⏳ Warm-up: Dando 3 segundos al OrderBook Streamer para estabilizar websockets...")
            await asyncio.sleep(3)

            logger.info(
                f"✅ WATCHLIST FINAL: {len(self.daily_watchlist)} pares "
                f"| 🔭 ANOMALY: {added_anomaly} | 🐋 WHALE: {added_whale}"
            )
            for p in self.daily_watchlist:
                layer = p.get('layer', 'ANOMALY')
                conf  = p.get('confidence', '?')
                logger.info(f"  [{layer:7s}][{conf:5s}] {p['symbol']:12s} ({p['direction']:5s}) | "
                            f"Score: {p['score']:3d} | {p.get('reasons','')[:70]}")


        except Exception as e:
            logger.error(f"Macro Scan Error: {e}")

    async def _get_btc_trend(self) -> str:
        """
        Determina la tendencia macro de BTC en el timeframe 15m.

        Returns:
            'BULL'    — BTC subiendo con fuerza (EMA20 > EMA50 + slope positivo)
            'BEAR'    — BTC bajando con fuerza (EMA20 < EMA50 + slope negativo)
            'NEUTRAL' — zona gris (EMA20 ≈ EMA50 o tendencia ambigua)
        """
        try:
            df = await self.market.get_klines('BTCUSDT', interval='15m', limit=60)
            if df is None or df.empty or len(df) < 52:
                return 'NEUTRAL'  # Sin datos — no bloqueamos

            close = df['close'].astype(float)
            ema20 = close.ewm(span=20, adjust=False).mean()
            ema50 = close.ewm(span=50, adjust=False).mean()

            last_ema20 = float(ema20.iloc[-1])
            last_ema50 = float(ema50.iloc[-1])
            last_price = float(close.iloc[-1])

            # Separación entre EMAs como % del precio
            spread_pct = (last_ema20 - last_ema50) / last_price * 100

            # Pendiente de las últimas 8 velas (velocidad de movimiento)
            slope = float(close.iloc[-1] - close.iloc[-8]) / float(close.iloc[-8]) * 100

            if spread_pct > 0.15 and slope > 0.3:    # EMA20 sobre EMA50 + subiendo
                return 'BULL'
            elif spread_pct < -0.15 and slope < -0.3: # EMA20 bajo EMA50 + bajando
                return 'BEAR'
            else:
                return 'NEUTRAL'

        except Exception as e:
            logger.debug(f"_get_btc_trend error: {e}")
            return 'NEUTRAL'  # Ante cualquier error, no bloqueamos

    async def whale_entry_loop(self):
        """
        Consumes señales de movimiento de ballena de la move_queue del WhaleWatcher.

        Cuando llega una señal (la ballena ya ejecutó su movimiento), este método:
        1. Verifica que el par no esté ya abierto y que haya capacidad
        2. 🦸 Filtro BTC: bloquea entradas que van fuerte contra la tendencia macro
        3. Descarga klines frescos y calcula ATR (requerido por execute_trade)
        4. Llama a execute_trade directamente — SIN confirm_entry

        Es el path de entrada exclusivo para pares WHALE. Los pares ANOMALY
        siguen usando el micro scan normal con confirm_entry.
        """
        logger.info("🐋 whale_entry_loop: iniciado — esperando señales de movimiento...")

        while self.running:
            try:
                # Esperar a que la queue esté inicializada (se crea en whale_watcher.start())
                if self.whale_watcher.move_queue is None:
                    await asyncio.sleep(1)
                    continue

                # Esperar la próxima señal de la queue (timeout para no bloquear)
                try:
                    whale_signal = await asyncio.wait_for(
                        self.whale_watcher.move_queue.get(), timeout=30.0
                    )
                except asyncio.TimeoutError:
                    continue

                sym       = whale_signal['symbol']
                direction = whale_signal['direction']

                logger.info(
                    f"🐋 WHALE ENTRY SIGNAL: {sym} {direction} | "
                    f"move_score={whale_signal['move_score']} | "
                    f"signals={whale_signal.get('move_signals', [])}"
                )

                # Verificar capacidad y que no esté ya abierto
                if len(self.market.positions) >= MAX_SIGNALS:
                    logger.info(f"⛔ WHALE ENTRY {sym}: capacidad llena ({MAX_SIGNALS}/{MAX_SIGNALS})")
                    continue
                # Verificar capacidad total y de layer WHALE
                if sym in self.market.positions:
                    logger.info(f"⚠️  WHALE ENTRY {sym}: posición ya abierta, ignorando")
                    continue
                whale_open = sum(1 for s, l in self.pos_layer.items()
                                 if l == 'WHALE' and s in self.market.positions)
                if whale_open >= WHALE_SLOTS:
                    logger.info(f"⛔ WHALE ENTRY {sym}: slots WHALE llenos ({whale_open}/{WHALE_SLOTS})")
                    continue
                if not self.blacklist.is_allowed(sym):
                    logger.info(f"🚫 WHALE ENTRY {sym}: en blacklist, ignorando")
                    continue
                if len(self.market.positions) >= ANOMALY_SLOTS + WHALE_SLOTS:
                    logger.info(f"⛔ WHALE ENTRY {sym}: máx posiciones totales alcanzado ({ANOMALY_SLOTS + WHALE_SLOTS})")
                    continue

                # ─── FILTRO BTC (tendencia macro) ─────────────────────────
                btc_trend   = await self._get_btc_trend()
                confidence  = whale_signal.get('confidence', 'LOW')
                move_score  = whale_signal.get('move_score', 0)

                # Bloquear si BTC va fuertemente en contra
                # Excepción: señales ULTRA con move_score muy alto pasan igual
                btc_blocks = (
                    (direction == 'LONG'  and btc_trend == 'BEAR') or
                    (direction == 'SHORT' and btc_trend == 'BULL')
                )
                ultra_override = (confidence == 'ULTRA' and move_score >= 120)

                if btc_blocks and not ultra_override:
                    logger.info(
                        f"🦸 WHALE ENTRY {sym} BLOQUEADA por BTC trend: "
                        f"dir={direction} vs BTC={btc_trend} | "
                        f"conf={confidence} move_score={move_score}"
                    )
                    continue

                if btc_blocks and ultra_override:
                    logger.info(
                        f"⚡ WHALE ULTRA OVERRIDE: {sym} entra contra BTC trend "
                        f"({btc_trend}) por señal extrema (move_score={move_score})"
                    )

                if btc_trend != 'NEUTRAL' and not btc_blocks:
                    logger.info(f"✅ BTC trend={btc_trend} alinea con {sym} {direction}")

                # Descargar klines y calcular indicadores (ATR requerido)
                try:
                    df = await self.market.get_klines(sym, interval='15m', limit=100)
                    if df is None or df.empty:
                        logger.warning(f"WHALE ENTRY {sym}: no hay datos")
                        continue

                    df_ind = await asyncio.to_thread(calculate_indicators, df)
                    if df_ind is None or df_ind.empty or 'atr' not in df_ind.columns:
                        logger.warning(f"WHALE ENTRY {sym}: falló el cálculo de ATR")
                        continue

                    # --- FILTRO ORDERBOOK ---
                    obi = self.ob_streamer.get_orderbook_imbalance(sym)
                    if (direction == 'LONG' and obi < -0.3) or (direction == 'SHORT' and obi > 0.3):
                        logger.warning(f"🛡️ WHALE OBI REJECT {sym} ({direction}): {obi:+.2f}")
                        continue
                        
                    wall_p, wall_dist = self.ob_streamer.get_nearest_wall(sym, direction)
                    if wall_dist is not None and wall_dist < 0.5:
                        logger.warning(f"🧱 WHALE WALL REJECT {sym} ({direction}): Muro a {wall_dist:.2f}%")
                        continue

                    # Señal directa (sin confirm_entry)
                    signal = {
                        'direction':     direction,
                        'score':         whale_signal.get('score', 0),
                        'strategy_mode': 'WHALE',
                        'layer':         'WHALE',
                        'reasons':       whale_signal.get('reasons', 'WHALE_MOVE'),
                    }
                    logger.info(
                        f"⚡ WHALE ENTRY: entrando en {sym} {direction} "
                        f"(score={signal['score']}, confianza={whale_signal.get('confidence','?')}) "
                        f"[slots WHALE: {whale_open+1}/{WHALE_SLOTS}]"
                    )
                    await self.execute_trade(sym, signal, df_ind)

                except Exception as e:
                    logger.error(f"WHALE ENTRY {sym}: error en entrada — {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"whale_entry_loop error: {e}")
                await asyncio.sleep(5)

    async def check_micro_entry(self, symbol, direction, pick_info):
        """Runs confirm_entry on the specific candidate with detailed logging."""
        try:
            # Verificar slots de ANOMALY antes de bajar klines
            anomaly_open = sum(1 for s, l in self.pos_layer.items()
                               if l == 'ANOMALY' and s in self.market.positions)
            total_open   = len(self.market.positions)
            if anomaly_open >= ANOMALY_SLOTS:
                logger.debug(f"⛔ ANOMALY slot lleno ({anomaly_open}/{ANOMALY_SLOTS}): {symbol} ignorado")
                return
            if total_open >= ANOMALY_SLOTS + WHALE_SLOTS:
                logger.debug(f"⛔ Posiciones totales al máx ({total_open}): {symbol} ignorado")
                return
            limit = 200
            df = await self.market.get_klines(symbol, interval='15m', limit=limit)

            if df.empty:
                logger.warning(f"📭 {symbol}: No klines data available")
                return

            df_indicators = await asyncio.to_thread(
                calculate_indicators,
                df
            )
            
            # Log indicator snapshot BEFORE confirm_entry
            curr = df_indicators.iloc[-2]  # confirm_entry uses iloc[-2]
            close_p = float(curr['close'])
            kf_price = float(curr['kf_price'])
            kf_slope = float(curr['kf_slope'])
            entropy_val = float(curr['entropy'])
            rsi_val = float(curr['rsi'])
            atr_val = float(curr['atr'])
            candle_color = 'GREEN' if close_p > float(curr['open']) else 'RED'
            
            is_valid = await asyncio.to_thread(
                confirm_entry,
                df_indicators,
                direction
            )
            
            if is_valid:
                # --- ORDERBOOK OBI & WALL FILTER ---
                obi = self.ob_streamer.get_orderbook_imbalance(symbol)
                if (direction == 'LONG' and obi < -0.3) or (direction == 'SHORT' and obi > 0.3):
                    logger.warning(f"🛡️ MICRO AVOIDANCE {symbol} ({direction}): Rechazo OBI {obi:+.2f}")
                    return
                
                wall_p, wall_dist = self.ob_streamer.get_nearest_wall(symbol, direction)
                if wall_dist is not None and wall_dist < 0.5:
                    logger.warning(f"🧱 MICRO AVOIDANCE {symbol} ({direction}): Muro a {wall_dist:.2f}%")
                    return

                logger.info(f"✨ MICRO TRIGGER: {symbol} ({direction}) Confirmed!")
                logger.info(f"   📈 Indicators: RSI={rsi_val:.1f} | Entropy={entropy_val:.3f} | KF_slope={kf_slope:.6f} | ATR={atr_val:.4f}")
                
                signal = {
                    'direction': direction,
                    'score': pick_info['score'],
                    'strategy_mode': 'TREND',
                    'layer':         pick_info.get('layer', 'ANOMALY'),
                    'reasons': pick_info['reasons']
                }
                anomaly_lbl = anomaly_open + 1
                logger.info(
                    f"✨ ANOMALY ENTRY: {symbol} ({direction}) confirmado! "
                    f"[slots ANOMALY: {anomaly_lbl}/{ANOMALY_SLOTS}] "
                    f"confianza={pick_info.get('confidence','?')}"
                )
                await self.execute_trade(symbol, signal, df_indicators)
                
        except Exception as e:
            logger.error(f"Micro Scan Error {symbol}: {e}")

    async def execute_trade(self, symbol, signal, df):
        """Execute trade with backtest-aligned sizing and state initialization."""
        curr_price = float(df['close'].iloc[-1])
        atr = float(df['atr'].iloc[-1])
        
        if atr <= 0: return
        
        # SL/TP Calculation (matches backtest)
        if signal['direction'] == 'LONG':
            sl = curr_price - (atr * INITIAL_SL_ATR)
            tp = curr_price + (atr * 10)
        else:
            sl = curr_price + (atr * INITIAL_SL_ATR)
            tp = curr_price - (atr * 10)
        
        # POSITION SIZING (exact backtest formula)
        equity = await self.get_equity()
        risk_amt = equity * RISK_PER_ENTRY
        risk_dist = abs(curr_price - sl)
        if risk_dist <= 0: return
        
        margin = (risk_amt / (risk_dist / curr_price)) / LEVERAGE
        margin = min(margin, equity * MAX_CAPITAL_PER_TRADE)
        amount = margin * LEVERAGE  # Convert to notional for MarketData
        
        if amount < 6.0: return
        
        logger.info(f"⚡ EXECUTING {symbol} | SL: {sl:.4f} | Size: ${amount:.0f} (Risk {RISK_PER_ENTRY*100}%)")
        
        result = await self.market.open_position(symbol, signal['direction'], amount, sl, tp)
        if result:
            layer = signal.get('layer', 'ANOMALY')
            logger.info(f"🔫 OPEN SUCCESS {symbol} [{layer}]")
            
            # Registrar layer de la posicion para tracking de slots por scanner
            self.pos_layer[symbol] = layer

            # Initialize local state tracker (matches backtest PositionManager)
            pid = PIDController(Kp=0.4, Ki=0.0, Kd=0.1, setpoint=0, output_limits=(-0.8, 0.5))
            self.pos_state[symbol] = {
                'best_price': curr_price,
                'be_locked': False,
                'pid': pid,
                'avg_price': curr_price,
                'atr_at_entry': atr,
                'scale_level': 1,
                'total_amount': margin,
                'entry_price': curr_price,
                'last_sl_update': 0,
                'failed_sl_count': 0,
            }


    async def scale_into_position(self, symbol, direction, notional):
        """Add to an existing position (scaling entry)."""
        try:
            if self.market.is_dry_run:
                # Mock: Update position amount directly
                pos = self.market.positions[symbol]
                price = await self.market.get_current_price(symbol)
                if price <= 0: return False
                new_qty = notional / price
                old_amount = pos['amount']
                pos['amount'] = old_amount + new_qty
                return True
            else:
                # Real: Place additional market order (Binance adds to position)
                price = await self.market.get_current_price(symbol)
                if price <= 0: return False
                qty = notional / price
                side_param = 'BUY' if direction == 'LONG' else 'SELL'
                info = await self.market._get_symbol_precision(symbol)
                qty_val = self.market._round_step_size(qty, info['q'])
                if qty_val <= 0: return False
                params = {
                    'symbol': symbol,
                    'side': side_param,
                    'type': 'MARKET',
                    'quantity': f"{qty_val}",
                }
                result = await self.market._signed_request('POST', '/fapi/v1/order', params)
                return result is not None
        except Exception as e:
            logger.error(f"Scale Error {symbol}: {e}")
            return False

    async def safety_loop(self):
        """Active Position Manager"""
        logger.info("Started Safety Monitor...")
        while self.running:
            try:
                await self.manage_positions()
                await asyncio.sleep(15)  # 15s interval (avoids API spam)
            except Exception as e:
                logger.error(f"Safety Loop Error: {e}")
                await asyncio.sleep(15)

    async def manage_positions(self):
        """
        Manages active positions CONCURRENTLY using PID Controller Logic.
        """
        current_tracked = self.ob_streamer.active_symbols
        active_positions = set(self.market.positions.keys())
        if not active_positions.issubset(current_tracked):
            await self.ob_streamer.update_focus_list(list(current_tracked | active_positions))

        tasks = [self._process_single_position(s, p) for s, p in list(self.market.positions.items())]
        if tasks:
            await asyncio.gather(*tasks)
            
        closed_symbols = [s for s in self.pos_state if s not in self.market.positions]
        for s in closed_symbols:
            self.pos_state.pop(s, None)
            self.pos_layer.pop(s, None)

    async def _process_single_position(self, symbol, pos):
        # Using if True to maintain the original 12-space indentation of the loop body
        if True:
            # --- 0. ORDER BOOK RADAR (ANTI-CRASH) ---
            wall_p, wall_dist = self.ob_streamer.get_nearest_wall(symbol, pos['side'])
            if wall_dist is not None and wall_dist < 0.75:
                logger.warning(f"🚨 ANTI-CRASH RADAR {symbol} {pos['side']}: Muro a {wall_dist:.2f}%. Huyendo!")
                await self.market.close_position(symbol, "ORDERBOOK_WALL_DETECTED")
                self.pos_state.pop(symbol, None)
                self.pos_layer.pop(symbol, None)
                return

            # Initialize state for orphaned positions (e.g., after bot restart)
            if symbol not in self.pos_state:
                logger.info(f"⚠️ Initializing state for orphaned position: {symbol}")
                pid = PIDController(Kp=0.4, Ki=0.0, Kd=0.1, setpoint=0, output_limits=(-0.8, 0.5))
                self.pos_state[symbol] = {
                    'best_price': pos['entry_price'],
                    'be_locked': False,
                    'pid': pid,
                    'avg_price': pos['entry_price'],
                    'atr_at_entry': 0,
                    'scale_level': 1,
                    'total_amount': pos.get('amount', 0) * pos['entry_price'] / LEVERAGE,
                    'entry_price': pos['entry_price'],
                    'last_sl_update': 0,
                    'failed_sl_count': 0,
                }
            
            state = self.pos_state[symbol]
            
            # 1. Get Fresh Market Data (ATR, Kalman, Price)
            try:
                df = await self.market.get_klines(symbol, interval='15m', limit=50)
                if df.empty: return
                
                df = await asyncio.to_thread(calculate_indicators, df)
                
                current_price = float(df['close'].iloc[-1])
                atr = float(df['atr'].iloc[-1])
                kf_price = float(df['kf_price'].iloc[-1])
                
                if current_price == 0: return
            except Exception as e:
                logger.error(f"Data Error {symbol}: {e}")
                return
            
            # ATR fallback (matches backtest)
            if atr <= 0:
                atr = state['atr_at_entry']
            if atr <= 0:
                atr = current_price * 0.02
            if state['atr_at_entry'] <= 0:
                state['atr_at_entry'] = atr
            
            # Track best price & calculate PnL (matches backtest exactly)
            if pos['side'] == 'LONG':
                if current_price > state['best_price']:
                    state['best_price'] = current_price
                pnl_atr = (current_price - state['avg_price']) / atr
                dist_kalman_atr = (current_price - kf_price) / atr
            else:
                if current_price < state['best_price']:
                    state['best_price'] = current_price
                pnl_atr = (state['avg_price'] - current_price) / atr
                dist_kalman_atr = (kf_price - current_price) / atr
            
            # --- 1. CHECK STOP LOSS ---
            is_sl = False
            if pos['side'] == 'LONG':
                if current_price <= pos['sl']: is_sl = True
            else:
                if current_price >= pos['sl']: is_sl = True
            
            if is_sl:
                reason = 'TRAILING_STOP' if state['be_locked'] else 'STOP_LOSS'
                logger.warning(f"🛑 {reason}: {symbol}")
                await self.market.close_position(symbol, reason)
                self.pos_state.pop(symbol, None)
                self.pos_layer.pop(symbol, None)
                return
            
            # --- 2. CHECK SCALING (before BE lock, matches backtest) ---
            if state['scale_level'] < MAX_SCALE and not state['be_locked']:
                ref_price = state['entry_price']
                if pos['side'] == 'LONG':
                    adverse_atr = (ref_price - current_price) / atr
                else:
                    adverse_atr = (current_price - ref_price) / atr
                
                should_scale = False
                if state['scale_level'] == 1 and adverse_atr >= SCALE_LEVEL_2:
                    should_scale = True
                elif state['scale_level'] == 2 and adverse_atr >= SCALE_LEVEL_3:
                    should_scale = True
                
                if should_scale:
                    equity = await self.get_equity()
                    risk_amt = equity * RISK_PER_ENTRY
                    risk_dist = abs(current_price - pos['sl'])
                    if risk_dist > 0:
                        new_margin = (risk_amt / (risk_dist / current_price)) / LEVERAGE
                        
                        # Enforce MAX_CAPITAL_PER_TRADE cumulatively across all entries
                        max_allowed_total = equity * MAX_CAPITAL_PER_TRADE
                        max_allowed_new = max(0.0, max_allowed_total - state['total_amount'])
                        new_margin = min(new_margin, max_allowed_new)
                        
                        notional = new_margin * LEVERAGE
                        if notional >= 5:
                            success = await self.scale_into_position(symbol, pos['side'], notional)
                            if success:
                                old_total = state['total_amount']
                                state['total_amount'] += new_margin
                                state['avg_price'] = (state['avg_price'] * old_total + current_price * new_margin) / state['total_amount']
                                state['scale_level'] += 1
                                logger.info(f"📈 SCALE {state['scale_level']}: {symbol} @ {current_price:.4f} | Avg: {state['avg_price']:.4f}")
            
            # --- 3. BREAKEVEN LOCK (matches backtest: BE_LOCK_ATR = 1.5) ---
            if not state['be_locked'] and pnl_atr >= BE_LOCK_ATR:
                buffer = state['avg_price'] * 0.002
                if pos['side'] == 'LONG':
                    new_sl = state['avg_price'] + buffer
                else:
                    new_sl = state['avg_price'] - buffer
                state['be_locked'] = True
                result = await self.market.update_sl(symbol, new_sl)
                state['last_sl_update'] = time.time()
                logger.info(f"🔒 BE LOCK: {symbol} SL->{new_sl:.4f} (PnL: {pnl_atr:.1f} ATR)")
            
            # --- 4. PID DYNAMIC TRAILING (after BE lock ONLY, matches backtest) ---
            if state['be_locked']:
                pid_adjust = state['pid'].update(dist_kalman_atr)
                
                # Trail = TRAIL_DISTANCE_ATR + PID output (matches backtest formula)
                current_trail_atr = TRAIL_DISTANCE_ATR + pid_adjust
                current_trail_atr = max(0.5, min(4.0, current_trail_atr))
                
                # Calculate desired SL
                if pos['side'] == 'LONG':
                    trail_sl = state['best_price'] - (atr * current_trail_atr)
                    should_update = trail_sl > pos['sl']
                else:
                    trail_sl = state['best_price'] + (atr * current_trail_atr)
                    should_update = trail_sl < pos['sl']
                
                if should_update:
                    # COOLDOWN: Don't spam SL updates (minimum 30s between updates)
                    elapsed_since_update = time.time() - state.get('last_sl_update', 0)
                    if elapsed_since_update < 30:
                        return
                    
                    # MINIMUM CHANGE: Only update if SL moves by at least 0.1%
                    change_pct = abs(trail_sl - pos['sl']) / pos['sl'] * 100 if pos['sl'] > 0 else 999
                    if change_pct < 0.1:
                        return
                    
                    await self.market.update_sl(symbol, trail_sl)
                    state['last_sl_update'] = time.time()
                    logger.info(f"🔄 PID TRAIL: {symbol} SL->{trail_sl:.4f} (Δ{change_pct:.2f}% | Adj:{pid_adjust:.2f})")
            
            # --- 5. MAX HOLD TIME (candle-equivalent via elapsed time) ---
            entry_time = pos.get('entry_time')
            if entry_time:
                elapsed_minutes = (datetime.now() - entry_time).total_seconds() / 60
                candles_elapsed = int(elapsed_minutes / 15)  # 15m candles
                if candles_elapsed >= MAX_HOLD_CANDLES:
                    logger.warning(f"⏰ MAX_HOLD_TIME: {symbol} ({candles_elapsed} candles)")
                    await self.market.close_position(symbol, "MAX_HOLD_TIME")
                    self.pos_state.pop(symbol, None)

    async def reporting_loop(self):
        """Periodically reports comprehensive status."""
        while self.running:
             try:
                 total_equity = self.market.balance
                 if not self.market.is_dry_run:
                     st = await self.market.get_real_account_status()
                     if st: total_equity = st['equity']
                 
                 open_count = len(self.market.positions)
                 wl_count = len(self.daily_watchlist)
                 pnl_pct = ((total_equity - self.start_balance) / self.start_balance * 100) if self.start_balance > 0 else 0
                 
                 logger.info(f"{'='*60}")
                 logger.info(f"📊 STATUS | Equity: ${total_equity:.2f} | Daily PnL: {pnl_pct:+.2f}% | Open: {open_count}/{MAX_SIGNALS} | Watchlist: {wl_count}")
                 
                 if open_count > 0:
                     for s, p in self.market.positions.items():
                         # Get current price for PnL calc
                         try:
                             curr_price = await self.market.get_current_price(s)
                         except:
                             curr_price = p['entry_price']
                         
                         # Calculate live PnL
                         if p['side'] == 'LONG':
                             live_pnl_pct = (curr_price - p['entry_price']) / p['entry_price'] * 100
                         else:
                             live_pnl_pct = (p['entry_price'] - curr_price) / p['entry_price'] * 100
                         
                         # Elapsed time
                         entry_time = p.get('entry_time')
                         if entry_time:
                             elapsed = datetime.now() - entry_time
                             elapsed_str = f"{elapsed.total_seconds()/3600:.1f}h"
                             candles = int(elapsed.total_seconds() / 900)  # 15m candles
                         else:
                             elapsed_str = "?h"
                             candles = 0
                         
                         # State info
                         state_str = ""
                         if s in self.pos_state:
                             st = self.pos_state[s]
                             be_icon = '✅' if st['be_locked'] else '❌'
                             state_str = (
                                 f"\n       BE:{be_icon} | Scale:{st['scale_level']}/{MAX_SCALE}"
                                 f" | Best:{st['best_price']:.4f} | Avg:{st['avg_price']:.4f}"
                                 f" | ATR@Entry:{st['atr_at_entry']:.4f}"
                             )
                         
                         logger.info(
                             f"   📌 {s} ({p['side']}) | Entry:{p['entry_price']:.4f} → Now:{curr_price:.4f}"
                             f" | PnL:{live_pnl_pct:+.2f}% | SL:{p['sl']:.4f}"
                             f" | Hold:{elapsed_str} ({candles}/{MAX_HOLD_CANDLES} candles)"
                             f"{state_str}"
                         )
                 else:
                     logger.info(f"   💤 No open positions")
                 
                 # Watchlist summary
                 if wl_count > 0:
                     wl_summary = ', '.join([f"{p['symbol']}({p['direction'][0]})" for p in self.daily_watchlist[:5]])
                     if wl_count > 5:
                         wl_summary += f" +{wl_count-5} more"
                     logger.info(f"   🎯 Watchlist: {wl_summary}")
                 
                 logger.info(f"{'='*60}")
                 
                 await asyncio.sleep(120) # 2 Minutes
             except Exception as e:
                 logger.error(f"Reporting Error: {e}")
                 await asyncio.sleep(60)
    
    async def get_equity(self):
        if not self.market.is_dry_run:
             s = await self.market.get_real_account_status()
             return s['equity'] if s else self.market.balance
        return self.market.balance

# ================================================================
# EXECUTION ENTRY POINT (Docker / Cloud Support)
# ================================================================
if __name__ == "__main__":
    # Load Environment Variables (Standard for Cloud)
    api_key = os.getenv('API_KEY', '')
    api_secret = os.getenv('API_SECRET', '')
    is_dry_run = os.getenv('DRY_RUN', 'true').lower() == 'true'

    if not api_key or not api_secret:
        if not is_dry_run:
            logging.error("❌ MISSING API KEYS! Set API_KEY and API_SECRET env vars.")
            sys.exit(1)
            
    try:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        bot = TrendBot(is_dry_run=is_dry_run, api_key=api_key, api_secret=api_secret)
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logging.error(f"Fatal Error: {e}")
        sys.exit(1)
