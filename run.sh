#!/bin/bash
set -e
cd "$(dirname "$0")"

if [ "$1" == "--install" ]; then
    python3 -m venv venv 2>/dev/null || true
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    echo "✅ Installed"
    exit 0
fi

if [ -d "venv" ]; then source venv/bin/activate; fi

case "$1" in
    --live)
        python3 main.py --live
        ;;
    --backtest)
        python3 main.py --backtest
        ;;
    *)
        echo "V8 FINAL PRODUCTION"
        echo "==================="
        echo "./run.sh --install    # Install"
        echo "./run.sh --backtest   # Test"
        echo "./run.sh --live       # Run"
        ;;
esac
