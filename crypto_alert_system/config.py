"""
config.py — Configuración del sistema de alertas.
Sin parámetros de trading. Solo escaneo y conexión.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # ── Binance API ────────────────────────────────────────────
    BINANCE_API_KEY    = os.getenv("BINANCE_API_KEY", "")
    BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")

    # ── Telegram ───────────────────────────────────────────────
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

    # ── Anomaly Scanner ────────────────────────────────────────
    ANOMALY_SCAN_INTERVAL_HOURS = 8    # Cada 8 horas en UTC
    ANOMALY_TOP_N               = 10   # Top 10 a reportar

    # ── Whale Scanner ──────────────────────────────────────────
    WHALE_TOP_N    = 15    # Pares a vigilar
    WHALE_MIN_SCORE = 130  # Score mínimo para incluir un par

    # ── Filtros de mercado ─────────────────────────────────────
    MIN_VOL_USDT_M = 5.0   # Volumen mínimo 24h en millones USDT

    # ── Sistema ────────────────────────────────────────────────
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    TIMEFRAME = "15m"


config = Config()
