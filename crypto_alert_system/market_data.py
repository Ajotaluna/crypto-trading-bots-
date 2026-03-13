"""
market_data.py — Obtención de datos de mercado desde Binance Futures.

Solo métodos de lectura. Sin ejecución de órdenes, sin gestión de posiciones.
"""
import asyncio
import logging
import os
import json
import time
from datetime import datetime

import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger("MarketData")


class MarketData:
    def __init__(self, api_key: str = "", api_secret: str = ""):
        self.api_key    = api_key
        self.api_secret = api_secret
        self.base_url   = "https://fapi.binance.com"
        self._cached_universe = None

        # Session con reintentos automáticos
        self.session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        self.session.mount('https://', HTTPAdapter(max_retries=retries))

    # ─────────────────────────────────────────────────────────────────────
    # UNIVERSO DE TRADING
    # ─────────────────────────────────────────────────────────────────────

    async def get_trading_universe(self):
        """
        Retorna la lista de pares a escanear.
        Prioridad:
         1. whitelist.json (si existe)
         2. Fallback: top 150 por volumen
        """
        if os.path.exists("whitelist.json"):
            try:
                with open("whitelist.json", "r") as f:
                    symbols = json.load(f)
                if symbols and len(symbols) > 0:
                    return symbols
            except Exception as e:
                logger.error(f"Failed to load whitelist.json: {e}")

        logger.warning("⚠️ whitelist.json no encontrado. Usando volumen dinámico.")
        return await self._scan_top_volume_fallback(limit=150)

    async def _scan_top_volume_fallback(self, limit: int = 150):
        """Fallback: obtiene los top pares por volumen 24h."""
        if self._cached_universe is not None:
            return self._cached_universe

        try:
            url = f"{self.base_url}/fapi/v1/ticker/24hr"

            def _fetch():
                return self.session.get(url, timeout=10)

            resp = await asyncio.to_thread(_fetch)

            if resp.status_code == 418:
                retry_after = int(resp.headers.get('Retry-After', 30))
                logger.warning(f"⚠️ 418 Ban — Retry-After: {retry_after}s")
                return []

            if resp.status_code == 200:
                data = resp.json()
                valid_pairs = []
                for x in data:
                    if not x['symbol'].endswith('USDT'):
                        continue
                    try:
                        vol = float(x['quoteVolume'])
                        if vol < 5_000_000:
                            continue
                        valid_pairs.append(x)
                    except Exception:
                        continue

                valid_pairs.sort(key=lambda x: float(x['quoteVolume']), reverse=True)
                self._cached_universe = [x['symbol'] for x in valid_pairs[:limit]]
                return self._cached_universe
            else:
                logger.warning(f"ticker/24hr status {resp.status_code}")
                return []
        except Exception as e:
            logger.error(f"Fallback Scan Failed: {e}")
            return []

    # ─────────────────────────────────────────────────────────────────────
    # KLINES (OHLCV)
    # ─────────────────────────────────────────────────────────────────────

    async def get_klines(self, symbol: str, interval: str = '15m',
                         limit: int = 100,
                         start_time=None, end_time=None) -> pd.DataFrame:
        """Descarga velas OHLCV de Binance Futures."""
        url = f"{self.base_url}/fapi/v1/klines"
        params = {'symbol': symbol, 'interval': interval, 'limit': limit}
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time

        def _fetch_and_parse():
            try:
                resp = self.session.get(url, params=params, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    df = pd.DataFrame(data, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_asset_volume', 'trades',
                        'taker_buy_base', 'taker_buy_quote', 'ignore'
                    ])
                    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'taker_buy_base']
                    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
                    df['taker_buy_vol'] = df['taker_buy_base']
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    return df
            except Exception as e:
                if "Connection" not in str(e):
                    logger.error(f"Error fetching klines for {symbol}: {e}")
            return pd.DataFrame()

        return await asyncio.to_thread(_fetch_and_parse)

    async def get_current_price(self, symbol: str) -> float:
        """Obtiene el precio actual del par."""
        tk = await self.get_klines(symbol, limit=1)
        if not tk.empty:
            return float(tk.iloc[-1]['close'])
        return 0.0

    # ─────────────────────────────────────────────────────────────────────
    # DATOS ON-CHAIN / FUTUROS
    # ─────────────────────────────────────────────────────────────────────

    async def get_open_interest(self, symbol: str, period: str = '1h'):
        """Obtiene el historial de Open Interest."""
        url = f"{self.base_url}/fapi/v1/openInterestHist"
        params = {'symbol': symbol, 'period': period, 'limit': 30}

        def _fetch():
            try:
                r = self.session.get(url, params=params, timeout=10)
                if r.status_code == 200:
                    return r.json()
            except Exception:
                pass
            return []

        return await asyncio.to_thread(_fetch)

    async def get_funding_rate(self, symbol: str, limit: int = 5):
        """Obtiene el historial de funding rate."""
        url = f"{self.base_url}/fapi/v1/fundingRate"
        params = {'symbol': symbol, 'limit': limit}

        def _fetch():
            try:
                r = self.session.get(url, params=params, timeout=5)
                if r.status_code == 200:
                    return [{'fundingRate': float(x['fundingRate']),
                             'fundingTime': int(x['fundingTime'])}
                            for x in r.json()]
            except Exception:
                pass
            return []

        return await asyncio.to_thread(_fetch)

    async def get_long_short_ratio(self, symbol: str, period: str = '1h', limit: int = 4):
        """Obtiene el ratio de cuentas long vs short."""
        url = f"{self.base_url}/futures/data/globalLongShortAccountRatio"
        params = {'symbol': symbol, 'period': period, 'limit': limit}

        def _fetch():
            try:
                r = self.session.get(url, params=params, timeout=5)
                if r.status_code == 200:
                    return [{'longShortRatio': float(x['longShortRatio']),
                             'longAccount':    float(x['longAccount']),
                             'shortAccount':   float(x['shortAccount'])}
                            for x in r.json()]
            except Exception:
                pass
            return []

        return await asyncio.to_thread(_fetch)

    async def get_agg_trades(self, symbol: str, limit: int = 500):
        """Obtiene los últimos trades agregados del par."""
        url = f"{self.base_url}/fapi/v1/aggTrades"
        params = {'symbol': symbol, 'limit': limit}

        def _fetch():
            try:
                r = self.session.get(url, params=params, timeout=5)
                if r.status_code == 200:
                    return [{'p': float(x['p']), 'q': float(x['q']),
                             'm': bool(x['m']), 't': int(x['T'])}
                            for x in r.json()]
            except Exception:
                pass
            return []

        return await asyncio.to_thread(_fetch)
