import os
import sys
import pandas as pd
import numpy as np

# Añadir el propio directorio al path para importaciones seguras
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scanner_anomaly import AnomalyScanner
from trading_strategy import calculate_indicators

def analyze_timing(data_dir: str, top_n: int = 10, scan_interval_hours: int = 24):
    print("=" * 80)
    print(" ⏱️ ANALIZADOR DE TIMING: ¿A qué hora explotan y a qué hora mueren los trades?")
    print("=" * 80)
    
    if not os.path.exists(data_dir):
        print(f"❌ Error: El directorio {data_dir} no existe.")
        return

    files = [f for f in os.listdir(data_dir) if f.endswith("_15m.csv")]
    
    print(f"📦 Extrayendo historial de {len(files)} monedas...")
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
    
    total_trades = 0
    
    # Listas para almacenar cuánto tiempo tomó llegar al Pico y al Suelo
    peak_times_15m = [] # Cantidad de velas 15m para llegar al MFE (Highest)
    bottom_times_15m = [] # Cantidad de velas 15m para llegar al MAE (Lowest)
    
    print(f"⚙️ Evaluando días simulados desde {start_time.date()} a {end_time.date()}...")
    
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
            
        for p in picks:
            sym = p['symbol']
            dir = p['direction']
            
            df = aligned_data[sym]
            mask = df['timestamp'] <= current_time
            if not mask.any(): continue
            t_idx = mask[mask].index[-1]
            
            if t_idx + 1 >= len(df): continue
            
            entry_idx = t_idx + 1
            entry_price = float(df.iloc[entry_idx]['open'])
            if pd.isna(entry_price) or entry_price <= 0: continue
            
            # Variables de seguimiento de Trade individual
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
                    
                # Si encontramos un nuevo pico ganador... grabamos el instante
                if mfe_now > highest_mfe:
                    highest_mfe = mfe_now
                    peak_candle_index = i
                    
                # Si encontramos un nuevo suelo perdedor (regresión mas profunda)...
                if mae_now < lowest_mae:
                    lowest_mae = mae_now
                    bottom_candle_index = i
            
            # Solo guardamos si realmente tuvo algún movimiento válido
            if peak_candle_index > 0:
                peak_times_15m.append(peak_candle_index)
                bottom_times_15m.append(bottom_candle_index)
                total_trades += 1
                
        current_time += delta_scan

    # --- CÁLCULO ESTADÍSTICO ---
    print("\n" + "=" * 80)
    print(" 📊 RESULTADOS: DISTRIBUCIÓN DE TIEMPOS DE VIDA")
    print("=" * 80)
    
    if total_trades == 0:
        print("❌ Cero trades útiles para analizar.")
        return
        
    print(f"Trades Top analizados     : {total_trades}")
    
    # 1. Análisis del Pico (Highest MFE)
    peak_series = pd.Series(peak_times_15m) * 15 / 60 # Convertimos índices de 15m a Horas
    
    print("\n🚀 ¿CUÁNDO ALCANZAN SU GANANCIA MÁXIMA (EL PICO)?")
    print(f"• Horas promedio hasta el pico : {peak_series.mean():.1f} horas")
    print(f"• Horas más frecuentes (Mediana): {peak_series.median():.1f} horas")
    print(f"• 25% explotan tempranamente en: menos de {peak_series.quantile(0.25):.1f} horas")
    print(f"• 75% logran su pico antes de  : {peak_series.quantile(0.75):.1f} horas")
    
    # 2. Análisis del Desplome o Regresión Máxima (Lowest MAE)
    bottom_series = pd.Series(bottom_times_15m) * 15 / 60
    
    print("\n🩸 ¿CUÁNDO SUFREN SU REGRESIÓN MÁS PROFUNDA (EL SUELO)?")
    print(f"• Horas promedio cayeron al fondo: {bottom_series.mean():.1f} horas")
    print(f"• Horas más frecuentes (Mediana) : {bottom_series.median():.1f} horas")
    print(f"• 25% sufren retroceso profundo  : en sus primeras {bottom_series.quantile(0.25):.1f} horas")
    print(f"• 75% lo sufren antes de superar : las {bottom_series.quantile(0.75):.1f} horas")
    
    # 3. Franjas Horarias del Pico
    print("\n🕰️ DISTRIBUCIÓN DEL PICO POR BLOQUES HORARIOS:")
    h_0_6 = len(peak_series[(peak_series >= 0) & (peak_series <= 6)])
    h_6_12 = len(peak_series[(peak_series > 6) & (peak_series <= 12)])
    h_12_18 = len(peak_series[(peak_series > 12) & (peak_series <= 18)])
    h_18_24 = len(peak_series[(peak_series > 18) & (peak_series <= 24)])
    
    print(f"• Primeras 6 Horas  : {h_0_6} trades ({(h_0_6/total_trades*100):.1f}%)")
    print(f"• Entre Hora 6 y 12 : {h_6_12} trades ({(h_6_12/total_trades*100):.1f}%)")
    print(f"• Entre Hora 12 y 18: {h_12_18} trades ({(h_12_18/total_trades*100):.1f}%)")
    print(f"• Últimas 6 Horas   : {h_18_24} trades ({(h_18_24/total_trades*100):.1f}%)")
    print("\n" + "-" * 80)
    print("💡 CONCLUSIÓN RECOMENDADA: ")
    print("    Si los trades (Ej. el 70%) tienden a llegar a su pico en las primeras X horas")
    print("    y luego comienzan a devolver las ganancias hacia la hora 18, esto significa que")
    print("    un límite estricto de cierre (Time Stop) o tomar ganancias parciales en esa franja")
    print("    horaria aumentará masivamente el Win Rate de la estrategia spot.")
    print("-" * 80 + "\n")

if __name__ == "__main__":
    print("\n" + "-"*60)
    print("Elige qué datos usar para la estadística temporal:")
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

    analyze_timing(data_dir)
