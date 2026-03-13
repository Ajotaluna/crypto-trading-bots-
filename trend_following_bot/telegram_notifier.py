import os
import requests
import logging
from dotenv import load_dotenv

# Credenciales hardcodeadas para producción (no requiere .env)
TELEGRAM_BOT_TOKEN = "8712190958:AAHlWO_JXjEY3S87FhAZjahUxq-jxxBrk-w"
TELEGRAM_CHAT_ID = "INGRESA_TU_CHAT_ID_AQUI"

logger = logging.getLogger("TrendBot.Telegram")

def send_telegram_message(text: str):
    """
    Función core para enviar mensajes a Telegram.
    Valida internamente que el token exista y sea válido.
    """
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "INGRESA_TU_TOKEN_AQUI":
        return False
    if not TELEGRAM_CHAT_ID or TELEGRAM_CHAT_ID == "INGRESA_TU_CHAT_ID_AQUI":
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown"
    }

    try:
        response = requests.post(url, json=payload, timeout=5)
        if response.status_code == 200:
            return True
        else:
            logger.error(f"Error enviando a Telegram: {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error de conexión Telegram: {e}")
        return False

# ═══════════════════════════════════════════════════════════
# ALERTAS DE ESCÁNERES (MACRO ANALYZERS)
# ═══════════════════════════════════════════════════════════

def alert_anomaly_scan(picks: list):
    """Format and send the Top picks from the Anomaly Scanner."""
    if not picks:
        return
        
    mensaje = "🔭 *NUEVO SCAN ANOMALÍAS (Nacientes)* 🔭\n"
    mensaje += "----------------------------------------\n"
    
    for i, p in enumerate(picks[:10], 1): # Top 10 max
        icono = "🟢" if p['direction'] == 'LONG' else "🔴"
        mensaje += f"#{i} {icono} *{p['symbol']}*\n"
        mensaje += f"   ➤ Score: {p['score']:.1f}pts | Confianza: {p.get('confidence', '?')}\n"
        # Extract short reason (first 60 chars)
        reason = p.get('reasons', '')[:60]
        mensaje += f"   ➤ Msg: _{reason}_\n\n"
        
    send_telegram_message(mensaje)

def alert_whale_scan(picks: list):
    """Format and send the Top executable picks from the Whale Scanner."""
    if not picks:
        return
        
    mensaje = "🐋 *NUEVO SCAN BALLENAS (Movimientos)* 🐋\n"
    mensaje += "----------------------------------------\n"
    
    for i, p in enumerate(picks[:15], 1): # Top 15 max
        icono = "🟢" if p['direction'] == 'LONG' else "🔴"
        mensaje += f"#{i} {icono} *{p['symbol']}*\n"
        mensaje += f"   ➤ Score: {p['score']} | Confianza: {p.get('confidence', '?')}\n"
        reason = p.get('reasons', '')[:60]
        mensaje += f"   ➤ Msg: _{reason}_\n\n"
        
    send_telegram_message(mensaje)
