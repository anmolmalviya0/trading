# FORGE Trading System

A professional, institutional-grade trading signal system built for real market conditions.

## What This System Does

- **Real-time market data** via Binance WebSocket (≤200ms updates)
- **Multi-timeframe signal generation** (5m, 15m, 30m, 1h)
- **Rule-based, deterministic signals** with scoring system (0-100)
- **Conviction filtering** - mandatory gatekeeper for all signals
- **Forensic-grade backtesting** with walk-forward validation
- **Professional web dashboard** with live prices and clocks

## What This System Does NOT Do

- ❌ Auto-execute trades (produces Trade Cards only)
- ❌ Guarantee daily profits
- ❌ Promise specific win rates
- ❌ Use fake or simulated data
- ❌ Trade outside conviction filters

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download fresh data
python run.py --download

# Start live dashboard
python run.py --live

# Open browser: http://localhost:8501
```

## Commands

| Command | Description |
|---------|-------------|
| `python run.py --live` | Start web dashboard |
| `python run.py --backtest` | Run backtests on all assets |
| `python run.py --download` | Download fresh market data |
| `python run.py --validate` | Walk-forward validation |

## Configuration

All thresholds are configurable in `config.yaml`:

- `signals.threshold`: Minimum score to generate signal (default: 70)
- `signals.min_confirmations`: Minimum indicators agreeing (default: 2)
- `conviction.enabled`: Enable/disable conviction filter
- `risk.max_trades_per_day`: Circuit breaker limit
- `risk.max_drawdown_pct`: Maximum allowed drawdown

## Assets

| Symbol | Asset |
|--------|-------|
| BTCUSDT | Bitcoin |
| ETHUSDT | Ethereum |
| SOLUSDT | Solana |
| BNBUSDT | BNB |
| PAXGUSDT | Gold (Pax Gold) |

## Timeframes

- 5 minutes
- 15 minutes
- 30 minutes
- 1 hour

## Signal Types

1. **EMA Cross** (50/200 EMA crossover)
2. **RSI Reversal** (oversold/overbought reversal)
3. **Bollinger Squeeze** (volatility breakout)
4. **Divergence** (price-RSI divergence)
5. **Volume Spike** (high volume confirmation)

## Conviction Filters

Every signal must pass ALL of these checks:

1. **Spread Filter** - Asset-specific spread threshold
2. **Regime Filter** - No trading in range/extreme volatility
3. **Session Filter** - Active during London/NY sessions
4. **Slippage Check** - Estimated impact within limits
5. **Quality Check** - Score and confirmation thresholds

## Walk-Forward Validation

| Period | Use |
|--------|-----|
| 2021-2023 | Training |
| 2024 | Validation |
| 2025 | Testing |
| 2026 | Forward simulation |

## File Structure

```
marketforge/
├── config.yaml         # All configuration
├── run.py              # Entry point
├── requirements.txt
├── README.md
├── core/
│   ├── data.py         # Data ingestion
│   ├── features.py     # Technical indicators
│   ├── signals.py      # Signal generation
│   ├── conviction.py   # Gatekeeper filters
│   ├── backtest.py     # Backtesting engine
│   └── database.py     # SQLite storage
├── live/
│   └── engine.py       # Live inference
├── ui/
│   └── dashboard.py    # Web interface
└── data/
    └── (parquet files)
```

## Limitations (Honest)

1. **Order Book Data**: Binance public API doesn't provide real-time order book depth via free WebSocket. System uses estimated spreads.

2. **Gold History**: PAXG only exists since 2020 (~6 years of data).

3. **News Feed**: Live RSS feeds often block requests. System uses fallback headlines.

4. **No Auto-Execution**: This is a signal generation system, not an automated trading bot.

## License

For personal use only. Not financial advice.
