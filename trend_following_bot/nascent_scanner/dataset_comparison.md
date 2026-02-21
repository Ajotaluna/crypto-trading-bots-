# Análisis de Composición de Datos

## Resumen Ejecutivo
- **Carpeta `data_monthly` (Test)**: Contiene **~148 pares**.
- **Carpeta `data` (Producción)**: Contiene **~333 pares**.
- **Diferencia Clave**: La carpeta de Producción (`data`) incluye **~185 pares extra** que NO están en la de Test. Estos pares extra suelen ser monedas **más nuevas, más volátiles o de menor capitalización**, lo que explica la caída de rendimiento al usarlos.

---

## 1. Contenido de `data_monthly` (La Muestra "Buena")
**¿Son solo estables?**
**NO.** Esta carpeta incluye una mezcla de todo, pero parece estar "curada" o ser más antigua.
- **Top Caps**: BTC, ETH, SOL, BNB, XRP, ADA.
- **Memecoins**: `1000BONK`, `1000PEPE`, `1000FLOKI`, `MEME`.
- **Acciones Tokenizadas**: `AMZN`, `TSLA`, `MSTR`, `PLTR`, `COIN`.
- **Proyectos Nuevos (High Volatility)**: `PYTH`, `STRK`, `WLD`, `ORDI`.
- **Raros/Experimentales**: `0G`, `AZTEC`, `CLANKER`, `FARTCOIN`, `GIGGLE`.

**Conclusión**: Es una muestra representativa del mercado (Grandes + Memes + Nuevos), pero **limitada en cantidad**.

---

## 2. Contenido Extra en `data` (El "Wild West")
La carpeta de Producción tiene todo lo anterior, MÁS un grupo de activos que parecen ser **lanzamientos recientes o de muy baja liquidez** (Low Caps).
Estos activos extra son los sospechosos de generar las pérdidas finales (Slippage alto, movimientos erráticos).

**Ejemplos de Pares SOLO en `data` (No en Monthly):**
- `ACT` (The AI Prophecy - Muy volátil recién lanzada)
- `ANIME` (Token de nicho)
- `ASR` (Fan Token - Baja liquidez)
- `ALCH` (Alchemy Pay - Volátil)
- `BULLA`, `COLLECT`, `CUDIS`
- `FIGHT`, `FOGO`, `HOOD`
- `INTC` (Intel - Acción)
- `KITE`, `LA`, `LIGHT`
- `MILK`, `MON`
- `MUSDT`?

## Diagnóstico
El drop de rendimiento en la prueba de año completo se debe a que el bot está operando en estos **~185 pares adicionales** de la carpeta `data`.
Al ser monedas con menos volumen o historial más errático (muchas recién listadas), las estrategias de tendencia puras (Trend Following) suelen fallar por falta de inercia o manipulación.

**Recomendación:**
Si quieres resultados consistentes con la prueba `monthly`, debes filtrar estos pares "basura" o operar solo en la lista blanca de los Top 100-150.
