#!/bin/bash
set -e

# 1. Update and Install Dependencies
echo ">>> Installing Docker & Git..."
sudo apt-get update
sudo apt-get install -y docker.io git docker-compose

# 2. Start Docker Service
sudo systemctl start docker
sudo systemctl enable docker
# Attempt to add user to group (requires re-login to take effect usually)
sudo usermod -aG docker $USER || true

# 3. Clone/Pull Repository
REPO_URL="https://github.com/Ajotaluna/crypto-trading-bots-.git"
DIR_NAME="crypto-trading-bots-"

if [ -d "$DIR_NAME" ]; then
    echo ">>> Repo exists, pulling latest..."
    cd $DIR_NAME
    git pull
else
    echo ">>> Cloning repo..."
    git clone $REPO_URL
    cd $DIR_NAME
fi

# 4. Build Docker Image
echo ">>> Building Docker Image..."
sudo docker build -t crypto-bot .

echo "=========================================="
echo ">>> SETUP COMPLETE!"
echo "=========================================="
echo "Option A: Run with Docker Compose (Easier)"
echo "1. Create/Edit .env file with your keys:"
echo "   nano .env"
echo "2. Start the bot:"
echo "   sudo docker-compose up -d"
echo ""
echo "Option B: Run with Docker Manually"
echo "sudo docker run -d --restart=always --name trend-bot \\"
echo "  -e API_KEY='YOUR_BINANCE_API_KEY' \\"
echo "  -e API_SECRET='YOUR_BINANCE_SECRET' \\"
echo "  -v \$(pwd)/data_cache:/app/data_cache \\"
echo "  crypto-bot"
echo "=========================================="
