"""
Funciones matemáticas de Order Book para el Whale Tracker.
Agregadas a whale_math.py como extensión de funciones OB puras.
"""

# ============================================================
# ORDER BOOK MATH — Funciones puras (stateless)
# ============================================================

import numpy as np


def orderbook_imbalance(bids, asks, levels=10):
    """
    Calcula el imbalance de presión bid/ask en los primeros N niveles.

    Args:
        bids: list de [price, qty] ordenado de mayor a menor precio
        asks: list de [price, qty] ordenado de menor a mayor precio
        levels: número de niveles a analizar

    Returns:
        (imbalance_ratio, bid_volume, ask_volume)
        - imbalance_ratio > 0: presión compradora dominante
        - imbalance_ratio < 0: presión vendedora dominante
        - rango: [-1.0, 1.0]
    """
    try:
        bids_vol = sum(float(q) for _, q in bids[:levels])
        asks_vol = sum(float(q) for _, q in asks[:levels])
        total = bids_vol + asks_vol
        if total == 0:
            return 0.0, 0.0, 0.0
        imbalance = (bids_vol - asks_vol) / total
        return imbalance, bids_vol, asks_vol
    except Exception:
        return 0.0, 0.0, 0.0


def detect_large_walls(bids, asks, levels=20, multiplier=5.0):
    """
    Detecta paredes de órdenes institucionales (iceberg pasivo).

    Una pared es una orden límite cuyo tamaño es `multiplier` veces
    el promedio del resto de niveles. Estas son señales de soporte/resistencia
    institucional: una ballena está esperando a un precio específico.

    Args:
        bids: list de [price, qty]
        asks: list de [price, qty]
        levels: niveles a analizar
        multiplier: cuántas veces el promedio para ser "pared"

    Returns:
        {
          'bid_walls': list de {'price', 'qty', 'multiple'},
          'ask_walls': list de {'price', 'qty', 'multiple'},
          'has_bid_wall': bool,
          'has_ask_wall': bool,
        }
    """
    try:
        def find_walls(side, n):
            qtys = [float(q) for _, q in side[:n]]
            if not qtys:
                return []
            avg = np.mean(qtys)
            if avg == 0:
                return []
            walls = []
            for i, (p, q) in enumerate(side[:n]):
                qty = float(q)
                mult = qty / avg
                if mult >= multiplier:
                    walls.append({'price': float(p), 'qty': qty, 'multiple': round(mult, 1)})
            return walls

        bid_walls = find_walls(bids, levels)
        ask_walls = find_walls(asks, levels)

        return {
            'bid_walls': bid_walls,
            'ask_walls': ask_walls,
            'has_bid_wall': len(bid_walls) > 0,
            'has_ask_wall': len(ask_walls) > 0,
        }
    except Exception:
        return {'bid_walls': [], 'ask_walls': [], 'has_bid_wall': False, 'has_ask_wall': False}


def cvd_from_agg_trades(trades_buffer):
    """
    CVD real calculado desde un buffer de aggTrades de Binance.

    El aggTrade de Binance incluye el campo `m` (is_buyer_maker):
    - m=True: el comprador es el maker → SELL taker (bajista)
    - m=False: el vendedor es el maker → BUY taker (alcista)

    Args:
        trades_buffer: list de dicts con campos:
          {'qty': float, 'm': bool, 'p': float}  (price, qty, is_buyer_maker)

    Returns:
        (cvd_value: float, cumulative_series: list[float])
    """
    try:
        if not trades_buffer:
            return 0.0, []

        cumulative = []
        running = 0.0
        for t in trades_buffer:
            qty = float(t.get('q', t.get('qty', 0)))
            is_buyer_maker = t.get('m', False)
            # m=True → taker SELL (negativo); m=False → taker BUY (positivo)
            delta = -qty if is_buyer_maker else qty
            running += delta
            cumulative.append(running)

        return running, cumulative
    except Exception:
        return 0.0, []


def spoofing_score(order_book_snapshots, levels=5, min_appearances=2, max_life_s=3.0):
    """
    Detecta órdenes de spoofing: aparecen y desaparecen en <3 segundos.

    Compara snapshots consecutivos del book. Una orden que aparece
    con volumen grande (>3x avg) y desaparece antes de ser llenada
    es spoofing institucional para mover el precio en la dirección opuesta.

    Args:
        order_book_snapshots: list de dicts:
          {'ts': float (epoch s), 'bids': [[p,q],...], 'asks': [[p,q],...]}
        levels: niveles a monitorear
        min_appearances: cuántas veces debe aparecer/desaparecer para contar
        max_life_s: máximo tiempo de vida en segundos para ser spoof

    Returns:
        (bid_spoof_count: int, ask_spoof_count: int, spoof_signal: bool)
    """
    try:
        if len(order_book_snapshots) < 2:
            return 0, 0, False

        bid_spoof = 0
        ask_spoof = 0

        # Track pesos de nivel entre snapshots consecutivos
        for i in range(1, len(order_book_snapshots)):
            prev = order_book_snapshots[i - 1]
            curr = order_book_snapshots[i]
            dt = curr['ts'] - prev['ts']

            if dt <= 0 or dt > max_life_s:
                continue

            prev_bids = {round(float(p), 6): float(q) for p, q in prev.get('bids', [])[:levels]}
            curr_bids = {round(float(p), 6): float(q) for p, q in curr.get('bids', [])[:levels]}
            prev_asks = {round(float(p), 6): float(q) for p, q in prev.get('asks', [])[:levels]}
            curr_asks = {round(float(p), 6): float(q) for p, q in curr.get('asks', [])[:levels]}

            # Avg volume del book
            all_prev_q = list(prev_bids.values()) + list(prev_asks.values())
            avg_q = np.mean(all_prev_q) if all_prev_q else 1.0

            # Bids que desaparecieron y eran grandes
            for price, qty in prev_bids.items():
                gone = price not in curr_bids or curr_bids[price] < qty * 0.1
                big = qty > avg_q * 3.0
                if gone and big:
                    bid_spoof += 1

            # Asks que desaparecieron y eran grandes
            for price, qty in prev_asks.items():
                gone = price not in curr_asks or curr_asks[price] < qty * 0.1
                big = qty > avg_q * 3.0
                if gone and big:
                    ask_spoof += 1

        spoof_signal = bid_spoof >= min_appearances or ask_spoof >= min_appearances
        return bid_spoof, ask_spoof, spoof_signal
    except Exception:
        return 0, 0, False
