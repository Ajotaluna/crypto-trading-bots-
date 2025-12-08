# Crypto Trading Bots (Trend + Scalping)

## 1. INSTALACIÓN EN AWS (Nueva Instancia)
Sigue estos pasos si estás configurando un servidor nuevo (Ubuntu).

### Paso 1: Descargar e Instalar
Copia y pega este bloque en tu terminal de AWS:
```bash
wget https://raw.githubusercontent.com/Ajotaluna/crypto-trading-bots-/main/deploy.sh
chmod +x deploy.sh
./deploy.sh
```

### Paso 2: Verificar Conexión con Binance
Comprueba que tu servidor NO esté bloqueado por región (USA = Bloqueado):
```bash
# Reemplaza con tus claves reales
sudo docker run --rm \
  -e API_KEY='TU_API_KEY' \
  -e API_SECRET='TU_API_SECRET' \
  crypto-bot python check_keys.py
```
*Si dice "✅ Private API OK!", puedes continuar.*

---

## 2. EJECUTAR LOS BOTS

### Opción A: MODO PRUEBA (Recomendado al inicio)
- **Dinero Ficticio (Dry Run)**: No gasta saldo real.
- **Modo Relajado (Loose Mode)**: Entra más rápido para que veas que funciona.

**Scalping Bot (Rápido):**
```bash
sudo docker run -d --restart=always --name scalp-bot \
  -e BOT_TYPE='scalp' \
  -e DRY_RUN='true' \
  -e LOOSE_MODE='true' \
  -e API_KEY='TU_API_KEY' \
  -e API_SECRET='TU_API_SECRET' \
  crypto-bot python -u scalping_bot_v2/main.py
```

**Trend Bot (Tendencias):**
```bash
sudo docker run -d --restart=always --name trend-bot \
  -e BOT_TYPE='trend' \
  -e DRY_RUN='true' \
  -e LOOSE_MODE='true' \
  -e API_KEY='TU_API_KEY' \
  -e API_SECRET='TU_API_SECRET' \
  crypto-bot python -u trend_following_bot/main.py
```

### Opción B: MODO REAL (Dinero Real)
- Solo quita `DRY_RUN` y `LOOSE_MODE`.
- **IMPORTANTE:** Configura "Margen Aislado" (Isolated) en Binance antes de empezar.

```bash
# Ejemplo Scalping Real
sudo docker run -d --restart=always --name scalp-bot \
  -e BOT_TYPE='scalp' \
  -e API_KEY='TU_API_KEY' \
  -e API_SECRET='TU_API_SECRET' \
  crypto-bot python -u scalping_bot_v2/main.py
```

---

## 3. VER REPORTE Y LOGS
Los bots ahora son silenciosos y solo reportan cada 5 minutos o cuando operan.

```bash
# Ver Scalping Bot
sudo docker logs -f scalp-bot

# Ver Trend Bot
sudo docker logs -f trend-bot
```

### Comandos Útiles
- **Detener bot:** `sudo docker stop scalp-bot`
- **Borrar bot:** `sudo docker rm -f scalp-bot`
- **Actualizar código:**
  ```bash
  cd crypto-trading-bots-
  git pull
  sudo docker build -t crypto-bot .
  ```
