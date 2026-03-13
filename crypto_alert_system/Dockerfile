# ─── Base image ───────────────────────────────────────────────────────
FROM python:3.11-slim

# ─── Metadata ─────────────────────────────────────────────────────────
LABEL maintainer="crypto-alert-system"
LABEL description="Crypto Analysis & Notification System — No trading"

# ─── System deps ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# ─── Workdir ──────────────────────────────────────────────────────────
WORKDIR /app

# ─── Python deps (cacheable layer) ───────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─── Application code ─────────────────────────────────────────────────
COPY . .

# ─── Run ──────────────────────────────────────────────────────────────
CMD ["python", "-u", "main.py"]
