import os
import sys
import pandas as pd
import numpy as np

# Añadir el propio directorio al path para importaciones seguras
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scanner_anomaly import AnomalyScanner
from trading_strategy import calculate_indicators

def analyze_btc_regime(data_dir: str, top_n: int = 10, scan_interval_hours: int = 24):
    print("=" * 80)
    print(" 👑 DIRECTOR DE RÉGIMEN: Analizando Correlación con Bitcoin (MFE vs MAE)")
    print("=" * 80)
    
    if not os.path.exists(data_dir):
        print(f"❌ Error: El directorio {data_dir} no existe.")
        return

    files = [f for f in os.listdir(data_dir) if f.endswith("_15m.csv")]
    
    # Comprobar si existe la data de Bitcoin
    btc_exists = any("BTCUSDT" in f for f in files)
    if not btc_exists:
         print(f"❌ Error: No se encontró BTCUSDT_15m.csv en {data_dir}. Es obligatorio para medir el régimen.")
         return
         
    print(f"📦 Extrayendo historial de {len(files)} monedas + BTC...")
    symbol_data = {}
    master_timeline = set()
    
    for f in files:
        sym = f.replace("_15m.csv", "")
        try:
            df = pd.read_csv(os.path.join(data_dir, f))
            time_col = 'Start_Time' if 'Start_Time' in df.columns else 'timestamp'
            df[time_col] = pd.to_datetime(df[time_col])
            df.sort_values(time_col, inplace=True)
            df.drop_duplicates(subset=[time_col], inplace=True)
            df.columns = [c.lower() for c in df.columns]
            
            if len(df) > 500:
                df = calculate_indicators(df)
                df['symbol'] = sym 
                symbol_data[sym] = df.reset_index(drop=True)
                master_timeline.update(df[time_col.lower()])
        except Exception as e:
            continue

    timeline = sorted(list(master_timeline))
    full_index = pd.DatetimeIndex(timeline)
    
    aligned_data = {}
    for sym, df in symbol_data.items():
        time_col = 'start_time' if 'start_time' in df.columns else 'timestamp'
        df.set_index(time_col, inplace=True)
        aligned = df.reindex(full_index)
        aligned['timestamp'] = aligned.index
        aligned.reset_index(drop=True, inplace=True)
        aligned_data[sym] = aligned
        
    min_dates = []
    max_dates = []
    for sym, df in aligned_data.items():
        if not df.empty and 'timestamp' in df.columns:
            min_dates.append(df['timestamp'].min())
            max_dates.append(df['timestamp'].max())
            
    if not min_dates:
        return
        
    start_time = max(min_dates) + pd.Timedelta(days=10)
    end_time = min(max_dates)
    current_time = start_time
    delta_scan = pd.Timedelta(hours=scan_interval_hours)
    
    # --- MÉTRICAS DE CORRELACIÓN ---
    total_trades = 0
    
    # Cuando la Altcoin llegó a su PICO (Highest MFE), ¿Qué estaba haciendo BTC desde la apertura?
    btc_during_alt_peaks_pct = [] 
    alt_peaks_pct = [] # Para cruzar
    
    alt_pumped_while_btc_pumped = 0 # Cantidad
    alt_pumped_while_btc_dumped = 0 # Altcoin subió A PESAR de que BTC bajaba
    
    # Cuando la Altcoin llegó a su SUELO (Lowest MAE), ¿Qué estaba haciendo BTC?
    btc_during_alt_bottoms_pct = []
    alt_bottoms_pct = []
    
    alt_dumped_while_btc_dumped = 0 # Altcoin fue arrastrada por la sangría de BTC
    alt_dumped_while_btc_pumped = 0 # Altcoin se desplomó SOLA mientras BTC sí subía
    
    print(f"⚙️ Evaluando régimen simulado desde {start_time.date()} a {end_time.date()}...")
    
    while current_time < end_time:
        current_view = {}
        for sym, df in aligned_data.items():
            mask = df['timestamp'] <= current_time
            if mask.any():
                t_idx = mask[mask].index[-1]
                if t_idx >= 120: 
                    slice_df = df.iloc[t_idx-120:t_idx+1]
                    current_view[sym] = slice_df
                    
        # Escanear el Top 10 ESTRICTO (Score >= 60, Kick >= 1.2)
        picks = AnomalyScanner.score_universe(current_view, -1, top_n=top_n)
        
        # Filtro de Calidad
        if len(picks) < 5:
            current_time += delta_scan
            continue
            
        # BTC Baseline data
        btc_df = aligned_data["BTCUSDT"]
        btc_mask = btc_df['timestamp'] <= current_time
        if not btc_mask.any(): 
            current_time += delta_scan
            continue
            
        btc_t_idx = btc_mask[btc_mask].index[-1]
        if btc_t_idx + 1 >= len(btc_df):
             current_time += delta_scan
             continue
             
        btc_entry_idx = btc_t_idx + 1
        btc_entry_price = float(btc_df.iloc[btc_entry_idx]['open'])
            
        for p in picks:
            sym = p['symbol']
            dir = p['direction']
            
            # No calculamos la correlación de BTC consigo mismo
            if sym == "BTCUSDT": continue
            
            df = aligned_data[sym]
            mask = df['timestamp'] <= current_time
            if not mask.any(): continue
            t_idx = mask[mask].index[-1]
            
            if t_idx + 1 >= len(df): continue
            
            entry_idx = t_idx + 1
            entry_price = float(df.iloc[entry_idx]['open'])
            if pd.isna(entry_price) or entry_price <= 0: continue
            
            highest_mfe = -999.0
            lowest_mae = 999.0
            peak_candle_index = 0
            bottom_candle_index = 0
            
            # Recorrer las 24 horas (96 velas)
            for i in range(1, 97):
                actual_idx = entry_idx + i
                if actual_idx >= len(df): break
                
                candle = df.iloc[actual_idx]
                close_p = float(candle['close'])
                
                if dir == 'LONG':
                    mfe_now = (candle['high'] - entry_price) / entry_price * 100
                    mae_now = (candle['low'] - entry_price) / entry_price * 100
                else:
                    mfe_now = (entry_price - candle['low']) / entry_price * 100
                    mae_now = (entry_price - candle['high']) / entry_price * 100
                    
                if mfe_now > highest_mfe:
                    highest_mfe = mfe_now
                    peak_candle_index = i
                    
                if mae_now < lowest_mae:
                    lowest_mae = mae_now
                    bottom_candle_index = i
            
            # Solo guardamos si realmente tuvo algún movimiento válido
            if peak_candle_index > 0:
                # ¿Qué estaba haciendo BTC exactamente en esa misma vela donde la Altcoin tocó el cielo?
                btc_peak_idx = min(btc_entry_idx + peak_candle_index, len(btc_df)-1)
                btc_price_at_alt_peak = float(btc_df.iloc[btc_peak_idx]['close'])
                
                # Crecimiento de BTC desde la apertura de NUESTRA operación
                btc_pnl_at_peak = (btc_price_at_alt_peak - btc_entry_price) / btc_entry_price * 100
                
                # ¿Qué estaba haciendo BTC exactamente en la vela donde nuestra Altcoin se desplomó al fondo?
                btc_bottom_idx = min(btc_entry_idx + bottom_candle_index, len(btc_df)-1)
                btc_price_at_alt_bottom = float(btc_df.iloc[btc_bottom_idx]['close'])
                btc_pnl_at_bottom = (btc_price_at_alt_bottom - btc_entry_price) / btc_entry_price * 100
                
                # Acumular data estadística
                btc_during_alt_peaks_pct.append(btc_pnl_at_peak)
                alt_peaks_pct.append(highest_mfe)
                
                btc_during_alt_bottoms_pct.append(btc_pnl_at_bottom)
                alt_bottoms_pct.append(lowest_mae)
                
                if btc_pnl_at_peak > 0:
                    alt_pumped_while_btc_pumped += 1
                else:
                    alt_pumped_while_btc_dumped += 1
                    
                if btc_pnl_at_bottom < 0:
                    alt_dumped_while_btc_dumped += 1
                else:
                    alt_dumped_while_btc_pumped += 1
                
                total_trades += 1
                
        current_time += delta_scan

    # --- CÁLCULO ESTADÍSTICO ---
    print("\n" + "=" * 80)
    print(" 📊 RESULTADOS: ESTATUTO DE CORRELACIÓN CON EL REY (BTC)")
    print("=" * 80)
    
    if total_trades == 0:
        print("❌ Cero trades útiles para analizar.")
        return
        
    print(f"Trades Top Altcoins analizados: {total_trades}")
    
    # 1. Análisis en Momentos de Exito (Pico / Highest MFE)
    print("\n🚀 CUANDO LAS ALTCOINS TOCARON SU PICO MÁXIMO (MFE):")
    print(f"• ¿Estaba BTC dictando el momentum? : Sí, en el {(alt_pumped_while_btc_pumped/total_trades*100):.1f}% de las veces BTC también iba subiendo (+).")
    print(f"• ¿Altcoins rebeldes?               : El {(alt_pumped_while_btc_dumped/total_trades*100):.1f}% de las veces las Altcoins del escáner subieron A PESAR de que BTC sangraba.")
    print(f"• Crecimiento promedio de la Altcoin: +{np.mean(alt_peaks_pct):.2f}%")
    print(f"• Crecimiento promedio de BTC       : {np.mean(btc_during_alt_peaks_pct):+.2f}% (Poder de arrastre)")
    
    # 2. Análisis en Momentos de Fracaso (Suelo / Lowest MAE / Regresiones)
    print("\n🩸 CUANDO LAS ALTCOINS SE DESPLOMARON O REGRESARON (MAE):")
    print(f"• ¿La caída fue culpa del Rey?      : En el {(alt_dumped_while_btc_dumped/total_trades*100):.1f}% de las veces, la caída fuerte ocurrió precisamente porque BTC también estaba cayendo (-).")
    print(f"• ¿Caídas por muerte de anomalía?   : El {(alt_dumped_while_btc_pumped/total_trades*100):.1f}% de las regresiones ocurrieron SOLAS. La Altcoin se desplomó mientras BTC estaba pacífico o subiendo.")
    print(f"• Caída promedio de la Altcoin (MAE): {np.mean(alt_bottoms_pct):.2f}%")
    print(f"• Comportamiento promedio de BTC    : {np.mean(btc_during_alt_bottoms_pct):+.2f}% mientras la Alt caía")
    
    print("\n" + "-" * 80)
    print("💡 DIAGNÓSTICO DEL RÉGIMEN: ")
    print("   SI BTC acompañó a las Altcoins en > 60% de los Picos:")
    print("     -> El Escáner caza 'BTC Followers Beta'. Necesitas que BTC suba para ganar.")
    print("   SI BTC acompañó a las Altcoins en < 40% de los Picos:")
    print("     -> El Escáner caza Verdaderas Anomalías Independientes (Descorrelacionadas).")
    print("   ")
    print("   SI BTC es > 70% responsable de las caídas (MAE):")
    print("     -> Deberías colocar un 'Kill-Switch'. Si BTC cae X%, abortar todos los trades.")
    print("-" * 80 + "\n")

if __name__ == "__main__":
    print("\n" + "-"*60)
    print("Elige qué datos usar para medir el Régimen contra BTC:")
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

    analyze_btc_regime(data_dir)
