"""
main.py — Orchestrator del sistema de alertas cripto.

Dos componentes corriendo en paralelo:

  1. ANOMALY SCANNER (cada 8 horas en UTC):
     - Descarga klines de todos los pares del universo
     - Ejecuta AnomalyScanner.score_universe()
     - Envía el Top 10 a Telegram con la hora UTC del scan

  2. WHALE WATCHER (continuo, cada 2 min):
     - Ejecuta scan_whale_universe() al inicio y cada 8h
     - WhaleWatcher monitorea los pares y alerta cuando detecta movimiento fuerte
"""
import asyncio
import logging
import sys
from datetime import datetime, timezone

from config import config
from market_data import MarketData
from scanner_anomaly import AnomalyScanner
from whale_market_scanner import scan_whale_universe
from whale_watcher import WhaleWatcher
from telegram_notifier import (
    alert_anomaly_scan,
    alert_whale_scan_summary,
    alert_system_start,
)

# ─────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s UTC | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("alert_system.log", encoding="utf-8"),
    ],
)
# Silenciar loggers ruidosos
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

logger = logging.getLogger("AlertSystem")


# ─────────────────────────────────────────────────────────────────────
# ANOMALY SCANNER LOOP (cada 8h UTC)
# ─────────────────────────────────────────────────────────────────────

async def anomaly_scan_loop(market: MarketData):
    """
    Ejecuta el Anomaly Scanner cada 8 horas en horario UTC.
    Horas de scan: 00:00, 08:00, 16:00 UTC.
    """
    SCAN_HOURS_UTC = {0, 8, 16}

    logger.info("📡 Anomaly Scanner loop iniciado | scans en 00:00, 08:00, 16:00 UTC")

    while True:
        try:
            now_utc = datetime.now(timezone.utc)
            current_hour = now_utc.hour

            # Calcular minutos hasta el próximo scan
            next_scan_hour = min(
                (h for h in sorted(SCAN_HOURS_UTC) if h > current_hour),
                default=min(SCAN_HOURS_UTC) + 24
            )
            remaining_minutes = (next_scan_hour - current_hour) * 60 - now_utc.minute
            if remaining_minutes <= 0:
                remaining_minutes = 1

            logger.info(
                f"⏳ Próximo Anomaly Scan: {next_scan_hour:02d}:00 UTC "
                f"(en {remaining_minutes} min)"
            )
            await asyncio.sleep(remaining_minutes * 60)

            # Ejecutar scan
            await _run_anomaly_scan(market)

        except asyncio.CancelledError:
            logger.info("Anomaly Scanner loop cancelado.")
            break
        except Exception as e:
            logger.error(f"Anomaly Scanner loop error: {e}")
            await asyncio.sleep(60)


async def _run_anomaly_scan(market: MarketData):
    """Descarga datos y ejecuta el scanner de anomalías."""
    now_utc  = datetime.now(timezone.utc)
    utc_hour = now_utc.hour
    logger.info(f"🔭 Ejecutando Anomaly Scan — {utc_hour:02d}:00 UTC")

    try:
        # 1. Obtener universo
        symbols = await market.get_trading_universe()
        if not symbols:
            logger.warning("⚠️ Universo vacío — saltando scan de anomalías")
            return

        logger.info(f"   Descargando klines para {len(symbols)} pares...")

        # 2. Descargar klines en paralelo (semáforo para no banear la IP)
        sem   = asyncio.Semaphore(5)
        tasks = {}

        async def _fetch_one(sym):
            async with sem:
                df = await market.get_klines(sym, interval='15m', limit=120)
                if df is not None and not df.empty:
                    df.columns = [c.lower() for c in df.columns]
                    return sym, df
                return sym, None

        results = await asyncio.gather(*[_fetch_one(s) for s in symbols])
        pair_data = {sym: df for sym, df in results if df is not None}

        if not pair_data:
            logger.warning("⚠️ Sin datos de klines — saltando scan de anomalías")
            return

        logger.info(f"   {len(pair_data)} pares con datos. Analizando...")

        # 3. Ejecutar scanner
        picks = AnomalyScanner.score_universe(
            pair_data=pair_data,
            now_idx=120,
            top_n=config.ANOMALY_TOP_N,
            long_ratio=None,
            boot_mode=False,
        )

        logger.info(f"   ✅ Anomaly Scan completo: {len(picks)} señales detectadas")

        # 4. Enviar a Telegram
        alert_anomaly_scan(picks, utc_hour)

    except Exception as e:
        logger.error(f"Error en Anomaly Scan: {e}", exc_info=True)


# ─────────────────────────────────────────────────────────────────────
# WHALE WATCHER LOOP (continuo)
# ─────────────────────────────────────────────────────────────────────

async def whale_watcher_loop(market: MarketData, watcher: WhaleWatcher):
    """
    Ejecuta el Whale Market Scan al inicio y cada 8h.
    El WhaleWatcher corre continuamente entre scans.
    """
    RESCAN_INTERVAL_H = 8

    logger.info("🐋 Whale Watcher loop iniciado")

    while True:
        try:
            # Escanear el universo
            logger.info("🐋 Ejecutando Whale Market Scan...")
            whale_picks = await scan_whale_universe(
                market,
                top_n=config.WHALE_TOP_N,
                min_score=config.WHALE_MIN_SCORE,
            )

            if whale_picks:
                logger.info(f"   {len(whale_picks)} pares whale detectados")
                watcher.update_pairs(whale_picks)
                alert_whale_scan_summary(whale_picks)
            else:
                logger.info("   Sin pares whale en este scan")

            # Esperar 8 horas antes del próximo re-scan del universo
            await asyncio.sleep(RESCAN_INTERVAL_H * 3600)

        except asyncio.CancelledError:
            logger.info("Whale Scanner loop cancelado.")
            break
        except Exception as e:
            logger.error(f"Whale Scanner loop error: {e}")
            await asyncio.sleep(300)  # 5 min de pausa ante error


async def whale_monitor_loop(market: MarketData, watcher: WhaleWatcher):
    """
    Inicia el WhaleWatcher con los pares actuales y lo mantiene vivo.
    Se llama después del primer whale scan.
    """
    logger.info("👁️  Iniciando monitor de whale moves...")
    await watcher.start(watcher._pairs, market)


# ─────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────

async def main():
    logger.info("=" * 65)
    logger.info("  🚀 CRYPTO ALERT SYSTEM — INICIANDO")
    logger.info("=" * 65)

    # Crear instancia de market data (solo lectura)
    market = MarketData(
        api_key=config.BINANCE_API_KEY,
        api_secret=config.BINANCE_API_SECRET,
    )

    # Crear watcher vacío (se poplará con el primer whale scan)
    watcher = WhaleWatcher()

    # Notificar inicio
    alert_system_start()

    # Ejecutar el primer whale scan antes de arrancar los loops
    logger.info("🐋 Ejecutando primer Whale Scan de arranque...")
    try:
        whale_picks = await scan_whale_universe(
            market,
            top_n=config.WHALE_TOP_N,
            min_score=config.WHALE_MIN_SCORE,
        )
        if whale_picks:
            watcher.update_pairs(whale_picks)
            alert_whale_scan_summary(whale_picks)
            logger.info(f"   {len(whale_picks)} pares cargados en el watcher")
    except Exception as e:
        logger.error(f"Error en whale scan inicial: {e}")

    # Ejecutar primer anomaly scan inmediatamente
    logger.info("🔭 Ejecutando primer Anomaly Scan de arranque...")
    await _run_anomaly_scan(market)

    # Lanzar todos los loops en paralelo
    logger.info("✅ Todos los módulos activos. Entrando en modo de vigilancia continua.")
    await asyncio.gather(
        anomaly_scan_loop(market),
        whale_watcher_loop(market, watcher),
        watcher.start(watcher._pairs, market),
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Sistema detenido manualmente.")
