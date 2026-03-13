import os
import sys
import pandas as pd
from datetime import datetime

# Añadir el propio directorio al path para importaciones seguras
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scanner_anomaly import AnomalyScanner

# ═══════════════════════════════════════════════════════════
# 📱 CONFIGURACIÓN DE TELEGRAM
# ═══════════════════════════════════════════════════════════
# Ingresa aquí tus credenciales de @BotFather
TELEGRAM_BOT_TOKEN = "8712190958:AAHlWO_JXjEY3S87FhAZjahUxq-jxxBrk-w"
TELEGRAM_CHAT_ID = "INGRESA_TU_CHAT_ID_AQUI"

def send_daily_picks(data_dir: str, top_n: int = 10):
    print("=" * 70)
    print(" 📡 TELEGRAM PINGER: Buscando Anomalías Spot Diarias...")
    print("=" * 70)
    
    if not os.path.exists(data_dir):
        print(f"❌ Error: El directorio {data_dir} no existe.")
        return

    files = [f for f in os.listdir(data_dir) if f.endswith("_15m.csv")]
    if not files:
        print(f"❌ Error: No se encontraron archivos CSV en {data_dir}.")
        return

    print(f"📦 Analizando {len(files)} monedas...")
    
    # 1. Obtener los últimos datos reales disponibles de cada archivo
    current_view = {}
    for f in files:
        sym = f.replace("_15m.csv", "")
        try:
            df = pd.read_csv(os.path.join(data_dir, f))
            # Renombrar columnas a minusculas para el scanner
            df.columns = [c.lower() for c in df.columns]
            
            # Solo necesitamos las ultimas 120 velas (30 horas) reales
            if len(df) >= 120:
                slice_df = df.tail(120).copy()
                slice_df.reset_index(drop=True, inplace=True)
                current_view[sym] = slice_df
        except Exception as e:
            continue

    if not current_view:
        print("❌ Error: No se pudo extraer data válida para analizar.")
        return

    # 2. Escanear todo el mercado en busca de las señales
    # boot_mode=-1 (Simulamos un inicio fresco para encontrar anomalías nacientes)
    picks = AnomalyScanner.score_universe(current_view, boot_mode=-1, top_n=top_n)

    if len(picks) < 4:
        print(f"🛑 MERCADO DÉBIL: El escáner sólo encontró {len(picks)} señales aptas (Min Requerido: 5).")
        print("💡 No se enviará alerta a Telegram porque no hay calidad suficiente hoy.")
        return

    # 3. Formatear y Enviar el Broadcast a Telegram
    print(f"\n✅ {len(picks)} Anomalías encontradas. Preparando envío...")
    
    if TELEGRAM_BOT_TOKEN == "INGRESA_TU_TOKEN_AQUI" or TELEGRAM_CHAT_ID == "INGRESA_TU_CHAT_ID_AQUI":
        print("⚠️ ADVERTENCIA: No has configurado TELEGRAM_BOT_TOKEN ni TELEGRAM_CHAT_ID.")
        print("⚠️ Edita este archivo (telegram_pinger.py) para colocar tus credenciales.")
        print("⚠️ Las selecciones de hoy fueron:")
        for p in picks:
            print(f"   ➤ {p['symbol']} | Score: {p['score']} | Kick: {p.get('volume_kick', 1.0)}x")
        return
        
    exito = AnomalyScanner.broadcast_to_telegram(picks, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
    
    if exito:
        print("📱 Pinger finalizado. Revisa tu celular.")

if __name__ == "__main__":
    print("\n" + "-"*60)
    print("Elige de dónde leer los datos para el Broadcast:")
    print("1: Data Larga (data_monthly) - Mayor historial, estadística fuerte")
    print("2: Data Corta (data) - Estadística reciente")
    print("-" * 60)
    
    opcion = input("Ingrese 1 o 2 (Enter por defecto = 1): ").strip()
    
    if opcion == "2":
        data_dir = r"c:\Users\Ajota\Documents\Nueva carpeta\trend_following_bot\nascent_scanner\data"
        print("Usando Data Corta...\n")
    else:
        data_dir = r"c:\Users\Ajota\Documents\Nueva carpeta\trend_following_bot\nascent_scanner\data_monthly"
        print("Usando Data Larga...\n")

    send_daily_picks(data_dir)
