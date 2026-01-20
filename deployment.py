"""
PHASE 5: DEPLOYMENT SCRIPTS
===========================
Docker and deployment configuration.

Includes:
- Dockerfile
- docker-compose.yml
- run scripts
- environment template
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent

# === CREATE DOCKERFILE ===
dockerfile_content = '''
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create directories
RUN mkdir -p logs models/ensemble models/production datasets

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/ || exit 1

# Run the application
CMD ["python", "live_terminal.py"]
'''

# === CREATE DOCKER-COMPOSE ===
compose_content = '''
version: '3.8'

services:
  trading-terminal:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./datasets:/app/datasets
      - ./models:/app/models
      - ./logs:/app/logs
      - ./market_data:/app/../market_data
    environment:
      - TELEGRAM_TOKEN=${TELEGRAM_TOKEN}
      - TELEGRAM_CHAT_ID=${TELEGRAM_CHAT_ID}
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/"]
      interval: 30s
      timeout: 10s
      retries: 3

  watchdog:
    build: .
    command: python ops_monitor.py watch
    volumes:
      - ./logs:/app/logs
      - ./paper_trading.db:/app/paper_trading.db
    environment:
      - TELEGRAM_TOKEN=${TELEGRAM_TOKEN}
      - TELEGRAM_CHAT_ID=${TELEGRAM_CHAT_ID}
    restart: unless-stopped
    depends_on:
      - trading-terminal
'''

# === CREATE REQUIREMENTS ===
requirements_content = '''
# Core
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
lightgbm>=4.0.0
joblib>=1.3.0

# Web
fastapi>=0.100.0
uvicorn>=0.23.0
websockets>=11.0
aiohttp>=3.8.0

# Data
pyarrow>=13.0.0

# Monitoring (optional)
# prometheus-client>=0.17.0

# Visualization (optional)
# streamlit>=1.25.0
# plotly>=5.15.0

# Explainability (optional)
# shap>=0.42.0
'''

# === CREATE .ENV TEMPLATE ===
env_template = '''
# Telegram Alerts (optional but recommended)
TELEGRAM_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

# Trading Configuration
POSITION_SIZE=1000
MAX_DAILY_TRADES=10
MAX_DAILY_LOSS=100
MIN_CONFIDENCE=55
'''

# === CREATE RUN SCRIPT ===
run_script = '''#!/bin/bash
# Run the trading system

set -e

cd "$(dirname "$0")"

# Check for virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Check for required files
if [ ! -d "models/ensemble" ]; then
    echo "❌ Models not found. Run training first:"
    echo "   python train_ensemble.py"
    exit 1
fi

# Start the terminal
echo "🚀 Starting Institutional Trading Terminal..."
python live_terminal.py
'''

# === CREATE STOP SCRIPT ===
stop_script = '''#!/bin/bash
# Stop all trading processes

echo "🛑 Stopping all trading processes..."

# Kill terminal
pkill -f "live_terminal" 2>/dev/null || true

# Kill watchdog
pkill -f "ops_monitor" 2>/dev/null || true

echo "✅ All processes stopped"
'''

# === CREATE FILES ===
def create_deployment_files():
    """Create all deployment files"""
    
    print("📦 Creating deployment files...")
    
    # Dockerfile
    with open(BASE_DIR / 'Dockerfile', 'w') as f:
        f.write(dockerfile_content.strip())
    print("   ✅ Dockerfile")
    
    # docker-compose.yml
    with open(BASE_DIR / 'docker-compose.yml', 'w') as f:
        f.write(compose_content.strip())
    print("   ✅ docker-compose.yml")
    
    # requirements.txt
    with open(BASE_DIR / 'requirements.txt', 'w') as f:
        f.write(requirements_content.strip())
    print("   ✅ requirements.txt")
    
    # .env.template
    with open(BASE_DIR / '.env.template', 'w') as f:
        f.write(env_template.strip())
    print("   ✅ .env.template")
    
    # run.sh
    run_script_path = BASE_DIR / 'run.sh'
    with open(run_script_path, 'w') as f:
        f.write(run_script.strip())
    os.chmod(run_script_path, 0o755)
    print("   ✅ run.sh")
    
    # stop.sh
    stop_script_path = BASE_DIR / 'stop.sh'
    with open(stop_script_path, 'w') as f:
        f.write(stop_script.strip())
    os.chmod(stop_script_path, 0o755)
    print("   ✅ stop.sh")
    
    print("\n✅ Deployment files created!")


# === MAIN ===
if __name__ == "__main__":
    print("="*70)
    print("📦 PHASE 5: DEPLOYMENT CONFIGURATION")
    print("="*70)
    
    create_deployment_files()
    
    print("\n" + "="*70)
    print("🚀 DEPLOYMENT OPTIONS")
    print("="*70)
    
    print("""
1. LOCAL DEVELOPMENT:
   ./run.sh

2. DOCKER:
   docker-compose up -d

3. VPS DEPLOYMENT:
   - Copy files to VPS
   - Set up .env with Telegram credentials
   - Run: docker-compose up -d

4. MONITORING:
   python ops_monitor.py watch

5. KILL SWITCH:
   python ops_monitor.py kill "Emergency reason"
   python ops_monitor.py resume
""")
