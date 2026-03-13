import os
import sys
import pandas as pd
import numpy as np

# Añadir el propio directorio al path para importaciones seguras
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scanner_anomaly import AnomalyScanner
from trading_strategy import calculate_indicators

def analyze_scanner_pure_metrics(data_dir: str, scan_interval_hours: int = 24, top_n: int = 10):
    print("=" * 80)
    print(" 🔬 RADIOGRAFÍA PURA DE ANOMALY SCANNER (NO TRADING, SÓLO MÁXIMOS Y MÍNIMOS)")
    print("=" * 80)
    
    if not os.path.exists(data_dir):
        print(f"❌ Error: El directorio {data_dir} no existe.")
        return

    files = [f for f in os.listdir(data_dir) if f.endswith("_15m.csv")]
    if not files:
        print("❌ Error: No se encontraron archivos *_15m.csv.")
        return

    print(f"📦 Cargando y alineando datos de {len(files)} pares...")
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

    if not symbol_data:
        print("❌ Error: Ningún par pudo ser parseado.")
        return

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
        
    # Averiguar fecha mínima y máxima común
    min_dates = []
    max_dates = []
    for sym, df in aligned_data.items():
        if not df.empty and 'timestamp' in df.columns:
            min_dates.append(df['timestamp'].min())
            max_dates.append(df['timestamp'].max())
            
    if not min_dates:
        print("❌ Datos insuficientes tras el alineamiento.")
        return
        
    start_time = max(min_dates) + pd.Timedelta(days=10) # 10 días warmup
    end_time = min(max_dates)
    
    print(f"⚙️ Ventana de simulación global: {start_time.date()} a {end_time.date()}")
    
    current_time = start_time
    delta_scan = pd.Timedelta(hours=scan_interval_hours)
    
    all_metrics = [] # Data list
    
    print("\nIniciando Esculqueo Diario...")
    
    while current_time < end_time:
        current_day = current_time.date()
        print(f"\n{'='*40}\n📅 DÍA: {current_day}\n{'='*40}")
        
        # 1. ESCANEAR OPORTUNIDADES (Top N puro)
        current_view = {}
        for sym, df in aligned_data.items():
            mask = df['timestamp'] <= current_time
            if mask.any():
                t_idx = mask[mask].index[-1]
                if t_idx >= 120: 
                    slice_df = df.iloc[t_idx-120:t_idx+1]
                    current_view[sym] = slice_df
                    
        picks = AnomalyScanner.score_universe(current_view, -1, top_n=top_n)
        
        for rank, p in enumerate(picks, 1):
            sym = p['symbol']
            dir = p['direction']
            
            df = aligned_data[sym]
            mask = df['timestamp'] <= current_time
            if not mask.any(): continue
            t_idx = mask[mask].index[-1]
            punto_0 = float(df.iloc[t_idx]['close'])
            
            if t_idx + 1 >= len(df): continue
            
            # Vela Inmediatamente Siguiente es el punto de impacto exacto
            entry_idx = t_idx + 1
            entry_price = float(df.iloc[entry_idx]['open']) # Entrada bruta para el calculo MFE/MAE
            
            if pd.isna(entry_price): continue
            
            # Extraer el desempeño de las próximas 24 horas (96 velas de 15m) después del escaneo
            future_24h = df.iloc[t_idx + 1 : min(t_idx + 97, len(df))]
            
            mfe_val = 0.0
            mae_val = 0.0
            close_val = 0.0
            
            if len(future_24h) > 0 and not future_24h['close'].isna().all():
                final_close = float(future_24h.iloc[-1]['close'])
                if dir == 'LONG':
                    mfe_val = (future_24h['high'].max() - entry_price) / entry_price * 100
                    mae_val = (future_24h['low'].min() - entry_price) / entry_price * 100
                    close_val = (final_close - entry_price) / entry_price * 100
                else: 
                    mfe_val = (entry_price - future_24h['low'].min()) / entry_price * 100
                    mae_val = (entry_price - future_24h['high'].max()) / entry_price * 100
                    close_val = (entry_price - final_close) / entry_price * 100
                
            all_metrics.append({
                'Date': current_day,
                'Rank': rank,
                'Symbol': sym,
                'Direction': dir,
                'Entry_Price': entry_price,
                'MFE_24h_Pct': mfe_val,
                'MAE_24h_Pct': mae_val,
                'Close_24h_Pct': close_val
            })
            
            estado = "+" if close_val > 0 else ""
            print(f"#{rank:<2} [{dir}] {sym:<10} | MFE: +{max(0,mfe_val):05.2f}% | MAE: {min(0,mae_val):<6.2f}% | 24H Cl: {estado}{close_val:.2f}%")
            
        current_time += delta_scan

    if len(all_metrics) == 0:
        print("❌ No se generaron métricas. Revisa los datos.")
        return
        
    # --- ANÁLISIS FINAL Y REPORTES ---
    df_results = pd.DataFrame(all_metrics)
    
    print("\n" + "=" * 80)
    print(" 📊 RESULTADOS DEL COMPORTAMIENTO BRUTO ("+ str(top_n) +" MEJORES PARES)")
    print("=" * 80)
    
    print(f"Total Días Analizados : {df_results['Date'].nunique()}")
    print(f"Total Predicciones    : {len(df_results)}")
    print(f"\nMFE Promedio Global   : +{df_results['MFE_24h_Pct'].mean():.2f}% (Movimiento Máximo Promedio a Favor)")
    print(f"MAE Promedio Global   : {df_results['MAE_24h_Pct'].mean():.2f}% (Caída Promedio en Contra)")
    print(f"Cierre 24H Promedio   : {df_results['Close_24h_Pct'].mean():+.2f}% (Estado exacto de la moneda a las 24 hrs)")
    
    print("\n--- Desglose de Cierre Neto (A las 24H exactas) ---")
    print(f"Terminan en Positivo    : {(len(df_results[df_results['Close_24h_Pct'] > 0]) / len(df_results) * 100):.1f}%")
    print(f"Terminan en Negativo    : {(len(df_results[df_results['Close_24h_Pct'] < 0]) / len(df_results) * 100):.1f}%")
    
    print("\n--- Desglose de Probabilidad de Exito (Alcance) en 24H ---")
    print(f"LLegan a +2%  : {(len(df_results[df_results['MFE_24h_Pct'] >= 2]) / len(df_results) * 100):.1f}% de las operaciones.")
    print(f"LLegan a +5%  : {(len(df_results[df_results['MFE_24h_Pct'] >= 5]) / len(df_results) * 100):.1f}% de las operaciones.")
    print(f"LLegan a +10% : {(len(df_results[df_results['MFE_24h_Pct'] >= 10]) / len(df_results) * 100):.1f}% de las operaciones.")
    print(f"LLegan a +15% : {(len(df_results[df_results['MFE_24h_Pct'] >= 15]) / len(df_results) * 100):.1f}% de las operaciones.")
    
    print("\n--- Tolerancia al Dolor Requerida (Suficiencia MAE) ---")
    print(f"Operaciones 'Perfectas' (Caen < -0.5% en contra antes de subir) : {(len(df_results[df_results['MAE_24h_Pct'] >= -0.5]) / len(df_results) * 100):.1f}%")
    print(f"Operaciones Normales (Retrocenden entre -1% y -3%)              : {(len(df_results[(df_results['MAE_24h_Pct'] <= -1) & (df_results['MAE_24h_Pct'] > -3)]) / len(df_results) * 100):.1f}%")
    print(f"Operaciones Difíciles (Se hunden > -3% temporalmente)           : {(len(df_results[df_results['MAE_24h_Pct'] <= -3]) / len(df_results) * 100):.1f}%")
    
    csv_file = "radiografia_completa.csv"
    df_results.to_csv(csv_file, index=False)
    print(f"\n📁 Archivo detallado (Excel/CSV) con TODAS las {len(df_results)} métricas creado en: {os.path.abspath(csv_file)}")
    
    # Exportar listado en crudo por día para leer en texto.
    txt_file = "radiografia_dia_por_dia.txt"
    with open(txt_file, "w", encoding="utf-8") as f:
        for date, group in df_results.groupby("Date"):
            f.write(f"\n{'='*40}\n📅 DÍA: {date}\n{'='*40}\n")
            for _, row in group.iterrows():
                estado_cierre = "+" if row['Close_24h_Pct'] > 0 else ""
                f.write(f"#{row['Rank']:<2} [{row['Direction']}] {row['Symbol']:<10} | A Favor (MFE): +{max(0,row['MFE_24h_Pct']):05.2f}% | En Contra (MAE): {min(0,row['MAE_24h_Pct']):<6.2f}% | Cierre a 24H: {estado_cierre}{row['Close_24h_Pct']:.2f}%\n")
                
    print(f"📝 Archivo de texto legible día por día exportado a: {os.path.abspath(txt_file)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Análisis de Radiografía Pura")
    # Agregué el path de 'data' por defecto para que podamos ejecutarlo rápido sino se usa el prompt.
    print("\n" + "-"*60)
    print("Elige qué datos usar para extraer métricas puras:")
    print("1: Data Larga (data_monthly) - Mayor historial, más lento")
    print("2: Data Corta (data) - Historial reciente, más rápido")
    print("-" * 60)
    
    opcion = input("Ingrese 1 o 2 (Enter por defecto = 1): ").strip()
    
    if opcion == "2":
        data_dir = r"c:\Users\Ajota\Documents\Nueva carpeta\trend_following_bot\nascent_scanner\data"
        print("Usando Data Corta...\n")
    else:
        data_dir = r"c:\Users\Ajota\Documents\Nueva carpeta\trend_following_bot\nascent_scanner\data_monthly"
        print("Usando Data Larga...\n")

    analyze_scanner_pure_metrics(data_dir, top_n=10)
