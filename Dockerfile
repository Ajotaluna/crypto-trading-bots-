FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
# build-essential and wget are good for general compilation if needed
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Set Python path to ensure imports work correctly
ENV PYTHONPATH=/app

# Default Environment Variables (can be overridden)
ENV BOT_TYPE='trend'
ENV PYTHONUNBUFFERED=1

# Command to run the Trend Following Bot
CMD ["python", "-u", "trend_following_bot/main.py"]
