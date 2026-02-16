# Trend Following Bot

Este repositorio contiene un robot de trading automatizado de seguimiento de tendencias para Binance Futures.

## Características

*   **Estrategia:** Trend Following con gestión de riesgos (Titan) y filtros avanzados (Kalman, Entropía).
*   **Seguridad:** Gestión de riesgos centralizada (RiskManager) y topes de margen.
*   **Universo:** Whitelist validada de ~297 pares (`whitelist.json`).
*   **Despliegue:** Contenedorizado con Docker para fácil despliegue en cualquier VPS.

## 🚀 Instalación y Despliegue

### Requisitos Previos

*   Cuenta en Binance Futures.
*   API Key y Secret (con permisos de Futuros).
*   Servidor VPS (Ubuntu recomendado) o máquina local con Docker.

### 1. Clonar el repositorio

```bash
git clone https://github.com/Ajotaluna/crypto-trading-bots-.git
cd crypto-trading-bots-
```

### 2. Configuración (Variables de Entorno)

Crea un archivo `.env` en la raíz del proyecto para guardar tus claves de forma segura:

```env
API_KEY=tu_api_key_aqui
API_SECRET=tu_api_secret_aqui
DRY_RUN=false
# DRY_RUN=true para modo simulación (sin dinero real)
```

---

### Opción A: Ejecución con Docker (Recomendada)

**Paso 1: Construir la imagen**
```bash
docker build -t crypto-bot .
```

**Paso 2: Ejecutar el contenedor**
```bash
docker run -d --restart=always --name trend-bot \
  --env-file .env \
  -v $(pwd)/data_cache:/app/data_cache \
  crypto-bot
```

**Ver logs:**
```bash
docker logs -f trend-bot
```

**Detener:**
```bash
docker stop trend-bot
```

---

### Opción B: Ejecución Manual (Python)

Si prefieres no usar Docker:

1.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Ejecutar:**
    ```bash
    # Exportar variables primero (Linux/Mac)
    export API_KEY='tu_api_key'
    export API_SECRET='tu_api_secret'
    
    # Ejecutar en segundo plano con nohup
    nohup python trend_following_bot/main.py > bot.log 2>&1 &
    
    # Ver logs
    tail -f bot.log
    ```

---

## 📂 Estructura del Proyecto

*   `trend_following_bot/`: Código fuente principal.
    *   `main.py`: Punto de entrada y bucle principal.
    *   `trading_strategy.py`: Lógica de trading (Indicadores, Entradas, Riesgo).
    *   `market_data.py`: Interacción con Binance API.
    *   `config.py`: Configuración del bot.
    *   `whitelist.json`: Universo de pares permitidos.
*   `data_cache/`: Datos persistentes (calibración, estado).
*   `nascent_scanner/`: Herramientas de Backtesting e Investigación.
*   `Dockerfile`: Configuración de la imagen Docker.

## 🛠 Mantenimiento

**Actualizar a la última versión:**

```bash
git pull
# Si usas Docker, reconstruye la imagen:
docker build -t crypto-bot .
docker stop trend-bot && docker rm trend-bot
# Vuelve a ejecutar el comando 'docker run' de arriba
```
