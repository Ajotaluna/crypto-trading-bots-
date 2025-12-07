#!/bin/bash
set -e

# 1. Update and Install Dependencies
echo ">>> Installing Docker & Git..."
sudo apt-get update
sudo apt-get install -y docker.io git

# 2. Start Docker Service
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER

# 3. Clone/Pull Repository
# REPLACE WITH YOUR REPO URL IF DIFFERENT
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
# We use 'sudo' in case the group update hasn't taken effect for current session
sudo docker build -t crypto-bot .

echo "=========================================="
echo ">>> SETUP COMPLETE!"
echo "=========================================="
echo "To run your TREND BOT, use this command (fill in keys):"
echo "sudo docker run -d --restart=always --name trend-bot \\"
echo "  -e BOT_TYPE='trend' \\"
echo "  -e API_KEY='YOUR_BINANCE_API_KEY' \\"
echo "  -e API_SECRET='YOUR_BINANCE_SECRET' \\"
echo "  crypto-bot python trend_following_bot/main.py"
echo ""
echo "To run your SCALPING BOT:"
echo "sudo docker run -d --restart=always --name scalp-bot \\"
echo "  -e BOT_TYPE='scalp' \\"
echo "  -e API_KEY='YOUR_BINANCE_API_KEY' \\"
echo "  -e API_SECRET='YOUR_BINANCE_SECRET' \\"
echo "  crypto-bot python scalping_bot_v2/main.py"
echo "=========================================="
