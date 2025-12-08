@echo off
echo ==============================================
echo   SETUP LOCAL ENVIRONMENT (Windows)
echo ==============================================

echo 1. Creating Virtual Environment (.venv)...
python -m venv .venv

echo 2. Activating Environment...
call .venv\Scripts\activate.bat

echo 3. Upgrading PIP...
python -m pip install --upgrade pip

echo 4. Installing Dependencies...
pip install -r requirements.txt

echo.
echo ==============================================
echo   SETUP COMPLETE!
echo ==============================================
echo To start the bot:
echo   1. .venv\Scripts\activate
echo   2. $env:LOOSE_MODE='true'; python scalping_bot_v2/main.py
echo ==============================================
pause
