# Trend Following Bot

Este repositorio contiene un robot de trading automatizado de seguimiento de tendencias para Binance Futures.

## Características
- Estrategia de Trend Following con gestión de riesgos (Titan).
- Escáner de mercado global.
- Ejecución segura de órdenes.
- Contenedorizado con Docker para fácil despliegue.

---

## 🚀 Instalación y Despliegue

### Requisitos Previo
- Cuenta en Binance Futures.
- API Key y Secret (con permisos de Futuros).
- Servidor VPS (Ubuntu recomendado) o máquina local con Docker.

### 1. Instalación Rápida (AWS / Ubuntu)
Ejecuta el script de despliegue automático:
```bash
wget https://raw.githubusercontent.com/Ajotaluna/crypto-trading-bots-/main/deploy.sh
chmod +x deploy.sh
./deploy.sh
```

### 2. Configuración y Ejecución

#### Opción Recomendada: Docker Compose
1. Crea un archivo `.env` en la raíz del proyecto:
   ```bash
   API_KEY=tu_api_key_aqui
   API_SECRET=tu_api_secret_aqui
   # Opcional: DRY_RUN=true con dinero ficticio (por defecto es false/real si no se pone)
   ```

2. Arranca el bot:
   ```bash
   sudo docker-compose up -d
   ```

3. Ver logs:
   ```bash
   sudo docker-compose logs -f
   ```

#### Opción Manual: Docker Run
```bash
sudo docker run -d --restart=always --name trend-bot \
  -e API_KEY='TU_API_KEY' \
  -e API_SECRET='TU_API_SECRET' \
  -v $(pwd)/data_cache:/app/data_cache \
  crypto-bot
```

---

## 🛠 Comandos de Mantenimiento

**Detener el bot:**
```bash
sudo docker-compose down
# O si usaste docker run:
# sudo docker stop trend-bot
```

**Actualizar a la última versión:**
```bash
git pull
sudo docker-compose build
sudo docker-compose up -d
```

---

## 📂 Estructura del Proyecto
- `trend_following_bot/`: Código fuente principal.
- `data_cache/`: Datos persistentes (calibración, estado).
- `_legacy_archive/`: Código antiguo/archivado (no utilizado).
- `Dockerfile` y `docker-compose.yml`: Configuración de contenedorización.
