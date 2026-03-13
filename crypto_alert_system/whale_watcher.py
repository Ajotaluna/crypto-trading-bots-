"""
whale_watcher.py — Vigilante asíncrono de movimientos de ballena.

Monitorea los pares con señales de ballena y espera el momento en que la
ballena ejecuta el movimiento fuerte. Cuando lo detecta, envía una alerta
directamente a Telegram vía telegram_notifier.

NOTA: A diferencia del bot de trading, aquí NO se publica en ninguna queue
ni se ejecutan órdenes. La señal termina en Telegram.

Señales de movimiento (puntuación acumulativa):
    BREAKOUT    +40  precio > máx 4 velas anteriores
    CVD_FLIP    +30  CVD invierte dirección bruscamente
    VOL_SURGE   +25  volumen actual > 3x media 12h
    ACCEL       +20  vela actual > ±1.5% en 15m
    MOMENTUM    +15  RSI > 60 (LONG) ó < 40 (SHORT)
    Umbral: ≥ 100 pts → alerta inmediata
"""
import asyncio
import logging
import time
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np

from telegram_notifier import alert_whale_move

logger = logging.getLogger("WhaleWatcher")

# ===================================================================
# CONFIGURACIÓN
# ===================================================================

POLL_INTERVAL      = 120    # Segundos entre chequeos (2 min)
KLINES_FAST        = 20     # Velas para análisis de movimiento
MOVE_THRESHOLD     = 100    # Puntos mínimos para disparar alerta
ACCEL_THRESHOLD    = 2.0    # % de retorno mínimo en 1 vela para ACCEL
VOL_SURGE_MULT     = 4.0    # Multiplicador de vol para VOL_SURGE
BREAKOUT_LOOKBACK  = 4      # Velas hacia atrás para detectar BREAKOUT
RSI_PERIOD         = 14     # Período de RSI
COOLDOWN_MIN       = 15     # Minutos mínimos entre alertas del mismo par
MAX_AGE_HOURS      = 6      # Horas máximas que un par permanece en el watcher


# ===================================================================
# HELPERS MATEMÁTICOS
# ===================================================================

def _rsi(series: pd.Series, period: int = RSI_PERIOD) -> float:
    """RSI simple."""
    try:
        delta = series.diff().dropna()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / loss.replace(0, 1e-10)
        rsi   = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1]) if not rsi.empty else 50.0
    except Exception:
        return 50.0


def _cvd_direction(df: pd.DataFrame) -> int:
    """
    Detecta flip de CVD: retorna +1 (compra), -1 (venta), 0 (sin flip).
    """
    try:
        if 'taker_buy_vol' in df.columns and df['taker_buy_vol'].sum() > 0:
            sell   = df['volume'] - df['taker_buy_vol']
            delta  = df['taker_buy_vol'] - sell
        else:
            body  = df['close'] - df['open']
            rng   = (df['high'] - df['low']).replace(0, 1e-10)
            delta = df['volume'] * (body / rng)

        recent = delta.iloc[-3:].sum()
        prev   = delta.iloc[-6:-3].sum()

        if prev < 0 and recent > 0:
            return +1
        if prev > 0 and recent < 0:
            return -1
        return 0
    except Exception:
        return 0


def _vol_ratio(df: pd.DataFrame) -> float:
    """Ratio: vol de las últimas 2 velas vs media de las 12 anteriores."""
    try:
        vol = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
        if len(vol) < 14:
            return 1.0
        recent_vol = vol.iloc[-2:].mean()
        hist_vol   = vol.iloc[-14:-2].mean()
        return float(recent_vol / hist_vol) if hist_vol > 0 else 1.0
    except Exception:
        return 1.0


def _price_accel(df: pd.DataFrame) -> float:
    """Retorno porcentual de la última vela."""
    try:
        c = pd.to_numeric(df['close'], errors='coerce')
        o = pd.to_numeric(df['open'],  errors='coerce')
        if len(c) < 2 or o.iloc[-1] == 0:
            return 0.0
        return float((c.iloc[-1] - o.iloc[-1]) / o.iloc[-1] * 100)
    except Exception:
        return 0.0


def _is_breakout(df: pd.DataFrame, direction: str) -> bool:
    """
    LONG:  cierre > máximo de las últimas BREAKOUT_LOOKBACK velas (excluyendo la actual)
    SHORT: cierre < mínimo de las últimas BREAKOUT_LOOKBACK velas (excluyendo la actual)
    """
    try:
        n = BREAKOUT_LOOKBACK + 1
        if len(df) < n:
            return False
        curr_close = float(df['close'].iloc[-1])
        prev       = df.iloc[-n:-1]
        if direction == 'LONG':
            return curr_close > float(prev['high'].max())
        else:
            return curr_close < float(prev['low'].min())
    except Exception:
        return False


# ===================================================================
# DETECCIÓN DEL MOVIMIENTO
# ===================================================================

def _detect_move(df: pd.DataFrame, direction: str) -> Dict[str, Any]:
    """
    Analiza si hay un movimiento de ballena en curso.
    """
    move_score = 0
    signals    = []

    close = pd.to_numeric(df['close'], errors='coerce')
    curr_price = float(close.iloc[-1]) if len(close) > 0 else 0.0

    # 1. BREAKOUT / BREAKDOWN
    if _is_breakout(df, direction):
        move_score += 40
        signals.append("BREAKOUT")

    # 2. CVD FLIP
    cvd_dir = _cvd_direction(df)
    if direction == 'LONG' and cvd_dir == +1:
        move_score += 30
        signals.append("CVD_FLIP_BULL")
    elif direction == 'SHORT' and cvd_dir == -1:
        move_score += 30
        signals.append("CVD_FLIP_BEAR")

    # 3. VOL SURGE
    vol_r = _vol_ratio(df)
    if vol_r >= VOL_SURGE_MULT:
        move_score += 25
        signals.append(f"VOL_SURGE({vol_r:.1f}x)")
    elif vol_r >= 2.0:
        move_score += 10
        signals.append(f"VOL_HIGH({vol_r:.1f}x)")

    # 4. PRICE ACCEL
    accel = _price_accel(df)
    if direction == 'LONG' and accel >= ACCEL_THRESHOLD:
        move_score += 20
        signals.append(f"ACCEL_UP(+{accel:.1f}%)")
    elif direction == 'SHORT' and accel <= -ACCEL_THRESHOLD:
        move_score += 20
        signals.append(f"ACCEL_DOWN({accel:.1f}%)")

    # 5. MOMENTUM (RSI)
    rsi_val = _rsi(close)
    if direction == 'LONG' and rsi_val > 60:
        move_score += 15
        signals.append(f"RSI_BULL({rsi_val:.0f})")
    elif direction == 'SHORT' and rsi_val < 40:
        move_score += 15
        signals.append(f"RSI_BEAR({rsi_val:.0f})")

    return {
        'move_score': move_score,
        'signals':    signals,
        'price':      curr_price,
        'vol_ratio':  vol_r,
        'accel_pct':  accel,
        'rsi':        rsi_val,
    }


# ===================================================================
# WHALE WATCHER
# ===================================================================

class WhaleWatcher:
    """
    Vigilante asíncrono de movimientos de ballena.

    Cuando detecta un movimiento (move_score >= MOVE_THRESHOLD), envía
    una alerta a Telegram directamente. No depende de ninguna queue
    ni de un módulo de trading.
    """

    def __init__(self, move_threshold: int = MOVE_THRESHOLD):
        self.move_threshold = move_threshold
        self._pairs: List[Dict[str, Any]] = []
        self._running = False
        self._last_alert: Dict[str, float] = {}
        self._added_at:   Dict[str, float] = {}

    def update_pairs(self, whale_picks: List[Dict[str, Any]]):
        """
        Actualiza la lista de pares vigilados.
        Llamar después de cada whale scan.
        """
        now = time.time()
        current_syms = {p['symbol'] for p in self._pairs}
        new_syms     = {p['symbol'] for p in whale_picks}

        added = 0
        for pick in whale_picks:
            if pick['symbol'] not in current_syms:
                self._pairs.append(pick)
                self._added_at[pick['symbol']] = now
                added += 1

        removed = len(current_syms - new_syms)
        self._pairs = [p for p in self._pairs if p['symbol'] in new_syms]

        if added or removed:
            logger.info(
                f"🔭 WhaleWatcher actualizado: {len(self._pairs)} pares "
                f"(+{added} añadidos, -{removed} removidos)"
            )

    async def start(self, initial_pairs: List[Dict[str, Any]], market):
        """
        Inicia el loop de vigilancia. Corre indefinidamente.

        Args:
            initial_pairs: output de scan_whale_universe()
            market:        instancia de MarketData
        """
        self._running = True
        self._pairs   = list(initial_pairs)
        now = time.time()
        for p in self._pairs:
            self._added_at[p['symbol']] = now

        logger.info(f"👁️  WhaleWatcher iniciado | {len(self._pairs)} pares bajo vigilancia")

        while self._running:
            try:
                await self._poll_all(market)
                await asyncio.sleep(POLL_INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"WhaleWatcher loop error: {e}")
                await asyncio.sleep(POLL_INTERVAL)

    async def stop(self):
        self._running = False
        logger.info("🛑 WhaleWatcher detenido.")

    async def _poll_all(self, market):
        """Chequea todos los pares vigilados en paralelo."""
        now        = time.time()
        max_age_s  = MAX_AGE_HOURS * 3600
        active     = [
            p for p in self._pairs
            if now - self._added_at.get(p['symbol'], now) < max_age_s
        ]

        if not active:
            logger.debug("WhaleWatcher: sin pares activos")
            return

        logger.debug(f"👁️  Polling {len(active)} pares whale...")

        sem = asyncio.Semaphore(8)

        async def _check_one(pick: Dict):
            async with sem:
                await self._check_pair(pick, market)

        await asyncio.gather(*[_check_one(p) for p in active])

    async def _check_pair(self, pick: Dict[str, Any], market):
        """Descarga klines y evalúa si el movimiento ya ocurrió."""
        sym       = pick['symbol']
        direction = pick.get('direction', 'NEUTRAL')

        if direction == 'NEUTRAL':
            return

        last = self._last_alert.get(sym, 0)
        if time.time() - last < COOLDOWN_MIN * 60:
            return

        try:
            df = await market.get_klines(sym, interval='15m', limit=KLINES_FAST)
            if df is None or df.empty or len(df) < 10:
                return

            df.columns = [c.lower() for c in df.columns]
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

            move = _detect_move(df, direction)

            if move['move_score'] >= self.move_threshold:
                self._last_alert[sym] = time.time()
                self._fire_alert(sym, pick, move)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug(f"WhaleWatcher error {sym}: {e}")

    def _fire_alert(
        self,
        symbol: str,
        whale_pick: Dict[str, Any],
        move: Dict[str, Any],
    ):
        """Loggea la alerta y la envía a Telegram."""
        dir_emoji  = '🟢' if whale_pick.get('direction') == 'LONG' else '🔴'
        direction  = whale_pick.get('direction', '?')
        signals_str = ' + '.join(move['signals'])

        logger.warning(
            f"\n{'='*65}\n"
            f"🚨 WHALE MOVE DETECTED: {symbol} {dir_emoji}{direction}\n"
            f"   Price: {move['price']:.4f} ({move['accel_pct']:+.1f}%) | "
            f"Vol: {move['vol_ratio']:.1f}x | RSI: {move['rsi']:.0f}\n"
            f"   Signals: {signals_str}  [score={move['move_score']}]\n"
            f"{'='*65}"
        )
        print(
            f"\n{'='*65}\n"
            f"🚨 WHALE MOVE: {symbol} {direction} | {signals_str}\n"
            f"{'='*65}",
            flush=True
        )

        # Enviar alerta a Telegram (sin queue, sin trading)
        try:
            alert_whale_move(symbol, whale_pick, move)
        except Exception as e:
            logger.error(f"Error enviando alerta whale a Telegram: {e}")
