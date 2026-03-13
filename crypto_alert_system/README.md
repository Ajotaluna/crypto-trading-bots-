# Crypto Alert System

Sistema independiente de análisis y notificaciones de mercado cripto.  
**Sin funcionalidad de trading** — solo análisis y alertas por Telegram.

## ¿Qué hace?

| Módulo | Frecuencia | Descripción |
|---|---|---|
| **Anomaly Scanner** | Cada 8h (00:00, 08:00, 16:00 UTC) | Detecta tendencias nacientes en todo el mercado. Reporta el **Top 10** por Telegram. |
| **Whale Watcher** | Continuo (cada 2 min) | Vigila pares con señales institucionales. Alerta inmediatamente cuando detecta el movimiento fuerte. |

## Despliegue con Docker

### 1. Configurar variables de entorno

```bash
cp .env.example .env
# Edita .env con tus credenciales
```

### 2. Construir y lanzar

```bash
docker-compose up -d --build
```

### 3. Ver logs en tiempo real

```bash
docker-compose logs -f
```

### 4. Detener

```bash
docker-compose down
```

## Variables de entorno requeridas

| Variable | Descripción |
|---|---|
| `BINANCE_API_KEY` | API Key de Binance (**solo permisos de lectura**) |
| `BINANCE_API_SECRET` | API Secret de Binance |
| `TELEGRAM_BOT_TOKEN` | Token del bot de Telegram |
| `TELEGRAM_CHAT_ID` | ID del chat/canal donde enviar alertas |

> **Nota**: La API Key de Binance solo necesita permisos de **lectura de mercado**. No se requieren permisos de trading.

## Estructura del proyecto

```
crypto_alert_system/
├── main.py                  # Orchestrator principal
├── config.py                # Configuración (lee variables de entorno)
├── market_data.py           # Conexión a Binance Futures (solo lectura)
├── scanner_anomaly.py       # Detector de tendencias nacientes
├── whale_market_scanner.py  # Scanner global de actividad de ballenas
├── whale_math_core.py       # Funciones matemáticas de detección
├── whale_watcher.py         # Monitor en tiempo real de movimientos
├── telegram_notifier.py     # Sistema de notificaciones Telegram
├── whitelist.json           # Lista de pares a monitorear
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

## Ejecución local (sin Docker)

```bash
pip install -r requirements.txt
cp .env.example .env
# Editar .env con tus credenciales
python main.py
```
