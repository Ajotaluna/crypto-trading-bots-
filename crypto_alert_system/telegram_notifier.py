"""
telegram_notifier.py — Sistema central de notificaciones Telegram.

Envía alertas del scanner de anomalías (cada 8h) y del whale watcher (en tiempo real).
"""
import logging
import requests
from config import config

logger = logging.getLogger("TelegramNotifier")

TELEGRAM_BOT_TOKEN = config.TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID   = config.TELEGRAM_CHAT_ID


# ═══════════════════════════════════════════════════════════
# CORE
# ═══════════════════════════════════════════════════════════

def send_telegram_message(text: str) -> bool:
    """Función base para enviar mensajes a Telegram."""
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "INGRESA_TU_TOKEN_AQUI":
        logger.warning("Telegram BOT_TOKEN no configurado.")
        return False
    if not TELEGRAM_CHAT_ID or TELEGRAM_CHAT_ID == "INGRESA_TU_CHAT_ID_AQUI":
        logger.warning("Telegram CHAT_ID no configurado.")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id":    TELEGRAM_CHAT_ID,
        "text":       text,
        "parse_mode": "Markdown",
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            return True
        else:
            logger.error(f"Error Telegram API: {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error de conexión Telegram: {e}")
        return False


# ═══════════════════════════════════════════════════════════
# ALERTAS — ANOMALY SCANNER (cada 8h)
# ═══════════════════════════════════════════════════════════

def alert_anomaly_scan(picks: list, utc_hour: int) -> bool:
    """
    Envía el Top 10 del Anomaly Scanner a Telegram.

    Args:
        picks:    lista de candidatos del AnomalyScanner
        utc_hour: hora UTC del scan (sin marca de tiempo completa)
    """
    if not picks:
        return send_telegram_message(
            f"🔭 *SCAN ANOMALÍAS — {utc_hour:02d}:00 UTC*\n"
            "Sin señales destacadas en este scan."
        )

    mensaje = f"🔭 *TOP ANOMALÍAS — {utc_hour:02d}:00 UTC* 🔭\n"
    mensaje += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"

    for i, p in enumerate(picks[:10], 1):
        icono = "🟢" if p['direction'] == 'LONG' else "🔴"
        conf  = p.get('confidence', '?')
        score = p.get('score', 0)
        reason = p.get('reasons', '')[:70]

        mensaje += f"*#{i}* {icono} `{p['symbol']}`\n"
        mensaje += f"   ➤ Score: *{score}pts* | Conf: `{conf}`\n"
        mensaje += f"   ➤ _{reason}_\n\n"

    mensaje += "⚠️ _Anomalías detectadas — Solo observación, sin trading._"

    result = send_telegram_message(mensaje)
    if result:
        logger.info(f"✅ Top {len(picks)} anomalías enviadas a Telegram ({utc_hour:02d}:00 UTC)")
    return result


# ═══════════════════════════════════════════════════════════
# ALERTAS — WHALE WATCHER (en tiempo real)
# ═══════════════════════════════════════════════════════════

def alert_whale_move(symbol: str, whale_pick: dict, move: dict) -> bool:
    """
    Envía una alerta de movimiento de ballena en tiempo real.

    Args:
        symbol:     par detectado (ej. "SOLUSDT")
        whale_pick: dict del whale scanner con contexto acumulación
        move:       dict de _detect_move con señales del movimiento
    """
    direction  = whale_pick.get('direction', '?')
    dir_emoji  = '🟢' if direction == 'LONG' else '🔴'
    confidence = whale_pick.get('confidence', '?')
    whale_score_val = whale_pick.get('score', '?')

    signals_str = ' + '.join(move.get('signals', []))
    whale_reasons = whale_pick.get('whale_reasons', [])
    whale_ctx = ', '.join(whale_reasons[:3]) if whale_reasons else whale_pick.get('reasons', '')

    mensaje  = f"🚨 *WHALE MOVE DETECTED* 🚨\n"
    mensaje += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    mensaje += f"*{symbol}* {dir_emoji} `{direction}`\n\n"
    mensaje += f"💰 Precio: `{move.get('price', 0):.4f}` ({move.get('accel_pct', 0):+.1f}% vela)\n"
    mensaje += f"📊 Vol: `{move.get('vol_ratio', 0):.1f}x` avg | RSI: `{move.get('rsi', 0):.0f}`\n"
    mensaje += f"⚡ Señales: _{signals_str}_\n"
    mensaje += f"   Move score: `{move.get('move_score', 0)}`\n\n"
    mensaje += f"🐋 Contexto whale `[{confidence}|{whale_score_val}pts]`:\n"
    mensaje += f"   _{whale_ctx[:100]}_\n\n"
    mensaje += "⚠️ _Solo alerta de observación — sin operación._"

    result = send_telegram_message(mensaje)
    if result:
        logger.info(f"✅ Whale alert enviada: {symbol} {direction}")
    return result


def alert_whale_scan_summary(picks: list) -> bool:
    """
    Envía resumen de los pares whale que ahora están bajo vigilancia.
    """
    if not picks:
        return False

    mensaje  = f"🐋 *WHALE WATCHLIST ACTUALIZADA* 🐋\n"
    mensaje += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    mensaje += f"Vigilando *{len(picks)} pares* con señales de ballena:\n\n"

    for i, p in enumerate(picks[:15], 1):
        icono = "🟢" if p['direction'] == 'LONG' else "🔴"
        mensaje += f"*#{i}* {icono} `{p['symbol']}` — Score: `{p['score']}` [{p.get('confidence','?')}]\n"

    mensaje += "\n_Se alertará cuando ocurra el movimiento fuerte._"

    return send_telegram_message(mensaje)


def alert_system_start() -> bool:
    """Notifica que el sistema arrancó correctamente."""
    mensaje  = "🚀 *CRYPTO ALERT SYSTEM — ONLINE* 🚀\n"
    mensaje += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    mensaje += "✅ Anomaly Scanner: activo (cada 8h UTC)\n"
    mensaje += "✅ Whale Watcher: activo (tiempo real)\n"
    mensaje += "_Sin funcionalidad de trading — solo análisis._"
    return send_telegram_message(mensaje)
