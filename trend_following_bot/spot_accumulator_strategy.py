import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Añadir el propio directorio al path para importaciones seguras
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scanner_anomaly import AnomalyScanner
from trading_strategy import calculate_indicators

COMMISSION = 0.001  # 0.1% Binance Spot

def run_spot_accumulator(data_dir: str, top_n: int = 10, initial_capital: float = 1000.0, scan_interval_hours: int = 24, time_stop_hours: int = 12):
    print("=" * 70)
    print(f" 🎒 INICIANDO SIMULACIÓN: ESTRATEGIA SPOT ACUMULADOR (TS {time_stop_hours}H)")
    print("=" * 70)
    
    if not os.path.exists(data_dir):
        print(f"❌ Error: El directorio {data_dir} no existe.")
        return

    files = [f for f in os.listdir(data_dir) if f.endswith("_15m.csv")]
    if not files:
        print("❌ Error: No se encontraron archivos *_15m.csv.")
        return

    print(f"📦 Alineando el multiverso temporal de {len(files)} monedas...")
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
    
    # ── ESTADÍSTICAS Y CUENTA ───────────────────────────────────────────
    stats = {
        'initial_equity': initial_capital,
        'current_equity': initial_capital,
        'peak_equity': initial_capital,
        'max_drawdown_pct': 0.0,
        
        'total_scans': 0,
        'trades_executed': 0,
        
        'absorptions_triggered': 0,      # Cuantas veces un ganador canibalizó a un perdedor
        'total_compounded_usd': 0.0,     # Dinero re-inyectado
        
        'wins_24h': 0,
        'losses_24h': 0
    }
    
    # Log Diario Array
    log_report = []

    print(f"⚙️ Ventana: {start_time.date()} a {end_time.date()}")
    print(f"💰 Capital Inicial: ${initial_capital:.2f}\n")
    
    while current_time < end_time:
        stats['total_scans'] += 1
        current_day = current_time.date()
        daily_log = [f"\n{'='*60}", f"📅 DÍA: {current_day} | CAPITAL LIBRE: ${stats['current_equity']:.2f}", f"{'='*60}"]
        
        # 1. ESCANEAR EL TOP 10 BRUTO Y DIVIDIR EL CAPITAL
        current_view = {}
        for sym, df in aligned_data.items():
            mask = df['timestamp'] <= current_time
            if mask.any():
                t_idx = mask[mask].index[-1]
                if t_idx >= 120: 
                    slice_df = df.iloc[t_idx-120:t_idx+1]
                    current_view[sym] = slice_df
                    
        # Escaneamos con los filtros estrictos normales
        picks = AnomalyScanner.score_universe(current_view, -1, top_n=top_n)
        
        if len(picks) < 4:
            daily_log.append(f"🛑 DÍA DESCARTADO (Mercado Muerto): El escáner sólo encontró {len(picks)} señales. Mínimo requerido: 5. Dinero a salvo en Cash.")
            print("\n".join(daily_log))
            log_report.extend(daily_log)
            current_time += delta_scan
            continue
            
        # Cuánto dinero toca por moneda hoy
        bag_size_usd = stats['current_equity'] / len(picks)
        
        # Entramos a mercado (Apertura de vela inmediatamente siguiente)
        open_positions = []
        
        for rank, p in enumerate(picks, 1):
            sym = p['symbol']
            dir = p['direction'] # 'LONG' o 'SHORT'
            
            df = aligned_data[sym]
            mask = df['timestamp'] <= current_time
            if not mask.any(): continue
            t_idx = mask[mask].index[-1]
            
            if t_idx + 1 >= len(df): continue
            
            entry_idx = t_idx + 1
            entry_price = float(df.iloc[entry_idx]['open'])
            if pd.isna(entry_price) or entry_price <= 0: continue
            
            # Cobrar fee de entrada
            entry_fee = bag_size_usd * COMMISSION
            invested_usd = bag_size_usd - entry_fee
            
            qty = invested_usd / entry_price
            
            open_positions.append({
                'symbol': sym,
                'direction': dir,
                'entry_price': entry_price, # Precio original
                'current_qty': qty,         # Cantidad de monedas (es variable porque puede comer perdedores)
                'total_invested': invested_usd, # Dinero total real metido aquí a lo largo del dia
                'entry_idx': entry_idx,
                'status': 'ACTIVE',         # ACTIVE o ABSORBED
                'highest_mfe': 0.0,         # Para registro
                'tiers_unlocked': 0         # Escalones de +10% superados
            })
            stats['trades_executed'] += 1

        stats['current_equity'] = 0.0 # Billetera vacía, todo el dinero está invertido
        daily_log.append(f"📦 Posiciones Abiertas: {len(open_positions)} (Aprox ${bag_size_usd:.2f} c/u)")

        # 2. MOTOR INTRADÍA: RECORRER VELAS HASTA EL TIME STOP (Por defecto 12 Horas / 48 Velas)
        # Evaluamos simultáneamente toda la canasta
        
        limit_candles = int(time_stop_hours * 4) # 1 hora = 4 velas de 15m
        
        for i in range(1, limit_candles + 1): # De +1 al Time Stop
            # a) Actualizar los PnLs flotantes vela por vela
            for pos in open_positions:
                if pos['status'] != 'ACTIVE': continue
                
                sym = pos['symbol']
                df = aligned_data[sym]
                actual_idx = pos['entry_idx'] + i
                
                if actual_idx >= len(df): continue
                
                # Precio actual High/Low evalúa MFE/MAE de esta velita
                candle = df.iloc[actual_idx]
                close_p = float(candle['close'])
                
                if pos['direction'] == 'LONG':
                    mfe_now = (candle['high'] - pos['entry_price']) / pos['entry_price'] * 100
                    pnl_current = (close_p - pos['entry_price']) / pos['entry_price'] * 100
                    pos['current_price'] = close_p
                    pos['floating_pnl_pct'] = pnl_current
                else: # SHORT
                    mfe_now = (pos['entry_price'] - candle['low']) / pos['entry_price'] * 100
                    pnl_current = (pos['entry_price'] - close_p) / pos['entry_price'] * 100
                    pos['current_price'] = close_p
                    pos['floating_pnl_pct'] = pnl_current
                    
                pos['highest_mfe'] = max(pos['highest_mfe'], mfe_now)
            
            # b) Buscar Acumuladores que rompieron nuevos Tiers (Escalones del 10%)
            for pos in open_positions:
                if pos['status'] != 'ACTIVE': continue
                
                # Calcular el Tier actual del PnL flotante (Ej: 15% -> 1, 25% -> 2, 31% -> 3)
                if pos['floating_pnl_pct'] >= 10.0:
                    current_tier = int(pos['floating_pnl_pct'] // 10)
                    
                    # Si subió a un nuevo escalón que no había desbloqueado antes
                    while current_tier > pos['tiers_unlocked']:
                        pos['tiers_unlocked'] += 1
                        
                        # Cobrar el premio: Buscar a la víctima (el "Zombi" más cercano a 0%)
                        potenciales = [p for p in open_positions if p['status'] == 'ACTIVE' and p['symbol'] != pos['symbol']]
                        
                        if potenciales:
                            # Encontramos la posición más cerca de 0 (absoluto)
                            victima = min(potenciales, key=lambda x: abs(x['floating_pnl_pct']))
                            
                            # La ejecutamos a mercado
                            victima['status'] = 'ABSORBED'
                            sell_price = victima['current_price']
                            
                            if victima['direction'] == 'LONG':
                                notional_val = victima['current_qty'] * sell_price
                                fees = notional_val * COMMISSION
                                rescued_capital = notional_val - fees
                            else:
                                cost_buy_back = victima['current_qty'] * sell_price
                                profit_usd = victima['total_invested'] - cost_buy_back
                                rescued_capital = victima['total_invested'] + profit_usd
                                fees = cost_buy_back * COMMISSION
                                rescued_capital -= fees

                            # El capital rescatado se inyecta en el Acumulador ganador
                            buy_price = pos['current_price']
                            entry_fee_comp = rescued_capital * COMMISSION
                            net_injected = rescued_capital - entry_fee_comp
                            
                            extra_qty = net_injected / buy_price
                            pos['current_qty'] += extra_qty
                            pos['total_invested'] += net_injected
                            
                            stats['absorptions_triggered'] += 1
                            stats['total_compounded_usd'] += net_injected
                            daily_log.append(f"🩸 TIER {pos['tiers_unlocked']} (+{pos['tiers_unlocked']*10}%): [{pos['symbol']}] (+{pos['floating_pnl_pct']:.1f}%) devoró al Zombi [{victima['symbol']}] ({victima['floating_pnl_pct']:.1f}%) | ${rescued_capital:.2f} inyectados.")
                        else:
                            # Nadie más vivo, dejamos de subir tiers para absorber
                            break

        # 3. Time Stop Automático (Cierre Exacto de Supervivientes)
        # Liquidamos CADA posición que siga activa al alcanzar el Time Stop
        total_recuperado_dia = 0.0
        
        daily_log.append(f"⏱️ TIME STOP ACTIVADO ({time_stop_hours}H) - Cerrando Sobrevivientes...")
        
        for pos in open_positions:
            if pos['status'] != 'ACTIVE':
                continue # Ya fue absorbida a perdida durante el día
                
            sym = pos['symbol']
            df = aligned_data[sym]
            exit_idx = min(pos['entry_idx'] + limit_candles, len(df) - 1)
            close_price = float(df.iloc[exit_idx]['close'])
            
            pnl_pct = (close_price - pos['entry_price']) / pos['entry_price'] * 100 if pos['direction'] == 'LONG' else (pos['entry_price'] - close_price) / pos['entry_price'] * 100
            estado = "+" if pnl_pct > 0 else ""
            daily_log.append(f"⏳ Cierre TimeStop: [{pos['symbol']}] | Invertido: ${pos['total_invested']:.2f} | PnL: {estado}{pnl_pct:.2f}%")
            
            if pnl_pct > 0:
                stats['wins_24h'] += 1
            else:
                stats['losses_24h'] += 1
                
            if pos['direction'] == 'LONG':
                notional_return = pos['current_qty'] * close_price
                fees = notional_return * COMMISSION
                net_return = notional_return - fees
            else:
                cost_buy_back = pos['current_qty'] * close_price
                profit_usd = pos['total_invested'] - cost_buy_back
                net_return = (pos['total_invested'] + profit_usd) - (cost_buy_back * COMMISSION)
                
            total_recuperado_dia += net_return
            
        stats['current_equity'] += total_recuperado_dia
        
        # Drawdown Tracking 
        if stats['current_equity'] > stats['peak_equity']:
            stats['peak_equity'] = stats['current_equity']
        else:
            dd = (stats['peak_equity'] - stats['current_equity']) / stats['peak_equity'] * 100
            if dd > stats['max_drawdown_pct']:
                stats['max_drawdown_pct'] = dd
                
        daily_log.append(f"💰 Balance Final del Día: ${stats['current_equity']:.2f}")
        
        # Imprimir en vivo
        print("\n".join(daily_log))
        log_report.extend(daily_log)
        
        current_time += delta_scan

    # 4. REPORTE FINAL GENERAL
    print("\n" + "=" * 80)
    print(f" 🎒 REPORTE FINANCIERO: ESTRATEGIA SPOT ACUMULADOR (TIME STOP {time_stop_hours}H)")
    print("=" * 80)
    
    total_closed = stats['wins_24h'] + stats['losses_24h']
    
    print(f"Total Días Simulados : {stats['total_scans']}")
    print(f"Posiciones Abiertas  : {stats['trades_executed']}")
    print(f"Canibalizaciones     : {stats['absorptions_triggered']} veces (Total re-inyectado: ${stats['total_compounded_usd']:.2f})")
    print(f"\n--- Probabilidad Cierre de Seguridad (Sobrevivientes al Time Stop) ---")
    if total_closed > 0:
        print(f"Monedas Sobrevivientes con Ganancia : {stats['wins_24h']} ({(stats['wins_24h']/total_closed*100):.1f}%)")
        print(f"Monedas Sobrevivientes en Pérdida   : {stats['losses_24h']} ({(stats['losses_24h']/total_closed*100):.1f}%)")
        
    net_profit = stats['current_equity'] - stats['initial_equity']
    net_profit_pct = (net_profit / stats['initial_equity']) * 100
    
    print("\n💰 RESULTADOS FINANCIEROS NETOS")
    print(f"Capital Inicial       : ${stats['initial_equity']:.2f}")
    print(f"Capital Real Final    : ${stats['current_equity']:.2f}")
    estado_pnl = "+" if net_profit > 0 else ""
    print(f"Ganancia/Pérdida Neta : {estado_pnl}${net_profit:.2f} ({estado_pnl}{net_profit_pct:.2f}%)")
    print(f"Máximo Drawdown       : -{stats['max_drawdown_pct']:.2f}%")
    
    # Exportar Reporte Intradía Completo
    with open("reporte_acumulador_dias.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log_report))
    print("\n📁 Log financiero completo generado en: reporte_acumulador_dias.txt")


if __name__ == "__main__":
    print("\n" + "-"*60)
    print("Elige qué datos usar para la estrategia ACUMULADOR:")
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

    run_spot_accumulator(data_dir, initial_capital=1000.0)
