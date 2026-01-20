#!/usr/bin/env python3
"""
FORGE TRADING SYSTEM
====================
Unified entry point for all operations.

Usage:
    python run.py --live          # Start live dashboard
    python run.py --backtest      # Run backtests
    python run.py --download      # Download fresh data
    python run.py --validate      # Walk-forward validation
"""
import argparse
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def run_live():
    """Start the live dashboard"""
    import subprocess
    
    dashboard_path = Path(__file__).parent / "ui" / "dashboard.py"
    
    print("Starting Forge Trading System...")
    print("Dashboard: http://localhost:8501")
    print("Press Ctrl+C to stop")
    
    subprocess.run([
        "streamlit", "run", str(dashboard_path),
        "--server.port", "8501",
        "--server.headless", "true"
    ])


def run_backtest():
    """Run backtests on all assets"""
    from core import HistoricalData, Backtester, load_config
    
    config = load_config()
    loader = HistoricalData()
    bt = Backtester(config)
    
    print("=" * 60)
    print("FORGE TRADING SYSTEM - BACKTEST")
    print("=" * 60)
    
    results = []
    
    for symbol in config.get('assets', []):
        for tf in config.get('timeframes', []):
            df = loader.load(symbol, tf)
            
            if df.empty:
                print(f"⚠ No data: {symbol} {tf}")
                continue
            
            print(f"\nBacktesting {symbol} {tf}...")
            result = bt.run(df, symbol, tf)
            
            results.append({
                'symbol': symbol,
                'timeframe': tf,
                **result
            })
            
            print(f"  Trades: {result['total_trades']}")
            print(f"  Win Rate: {result['win_rate']:.1%}")
            print(f"  Profit Factor: {result['profit_factor']:.2f}")
            print(f"  Max DD: {result['max_drawdown']:.1%}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total_trades = sum(r['total_trades'] for r in results)
    avg_wr = sum(r['win_rate'] for r in results) / len(results) if results else 0
    avg_pf = sum(r['profit_factor'] for r in results) / len(results) if results else 0
    
    print(f"Total Backtests: {len(results)}")
    print(f"Total Trades: {total_trades}")
    print(f"Avg Win Rate: {avg_wr:.1%}")
    print(f"Avg Profit Factor: {avg_pf:.2f}")

    # Save results to JSON for dashboard
    import json
    backtest_data = {
        "summary": {
            "data_range": "2021-01-01 to 2026-01-19",
            "timeframe": "Multi",
            "total_trades": total_trades,
            "avg_win_rate": avg_wr,
            "avg_profit_factor": avg_pf
        },
        "assets": {},
        "confidence_score": int(avg_wr * 100),
        "honest_assessment": {
            "can_achieve_65_70_wr": avg_wr >= 0.65,
            "can_achieve_profitability": avg_pf > 1.0,
            "key_insight": "Institutional features (VWAP/OFI) active. Trade count lower but higher quality.",
            "recommendation": "Monitor forward testing.",
            "limitation": "Backtest period includes 2022 bear market."
        }
    }
    
    for r in results:
        key = r['symbol']
        backtest_data['assets'][key] = {
            "name": key,
            "trades": r['total_trades'],
            "win_rate": r['win_rate'],
            "profit_factor": r['profit_factor'],
            "total_pnl_pct": r['net_profit_pct'],
            "sample_entry": 0, # Not detailed in summary run
            "sample_tp": 0,
            "sample_sl": 0,
            "strategy": "Institutional (VWAP+OFI)",
            "note": f"WR: {r['win_rate']:.1%} PF: {r['profit_factor']:.2f}"
        }
        
    # Custom JSON Encoder for Numpy types
    import numpy as np
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.bool_):
                return bool(obj)
            return super(NumpyEncoder, self).default(obj)

    save_path = Path("data/per_asset_backtest.json")
    with open(save_path, "w") as f:
        json.dump(backtest_data, f, indent=2, cls=NumpyEncoder)
    print(f"\n✅ Results saved to {save_path}")


def run_download():
    """Download fresh data for all assets"""
    from core import HistoricalData
    
    print("=" * 60)
    print("FORGE TRADING SYSTEM - DATA DOWNLOAD")
    print("=" * 60)
    
    loader = HistoricalData()
    loader.download_all()
    
    print("\n✓ Data download complete")


def run_validate():
    """Run walk-forward validation"""
    from core import HistoricalData, WalkForwardValidator, load_config
    
    config = load_config()
    loader = HistoricalData()
    validator = WalkForwardValidator(config)
    
    print("=" * 60)
    print("FORGE TRADING SYSTEM - WALK-FORWARD VALIDATION")
    print("=" * 60)
    
    for symbol in config.get('assets', []):
        for tf in config.get('timeframes', []):
            df = loader.load(symbol, tf)
            
            if df.empty:
                print(f"⚠ No data: {symbol} {tf}")
                continue
            
            print(f"\nValidating {symbol} {tf}...")
            result = validator.validate(df, symbol, tf)
            
            for period, data in result['periods'].items():
                status = "✓" if data['passed'] else "⚠"
                print(f"  {period}: {status} WR={data['win_rate']:.1%} PF={data['profit_factor']:.2f}")
                
                for warn in data.get('warnings', []):
                    print(f"    ⚠ {warn}")
            
            overall = "PASS" if result['valid'] else "FAIL"
            print(f"  Overall: {overall} ({result['passed_count']}/{result['total_periods']} periods)")


def main():
    parser = argparse.ArgumentParser(
        description="Forge Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run.py --live          Start live dashboard
    python run.py --backtest      Run backtests
    python run.py --download      Download fresh data
    python run.py --validate      Walk-forward validation
        """
    )
    
    parser.add_argument('--live', action='store_true', help='Start live dashboard')
    parser.add_argument('--backtest', action='store_true', help='Run backtests')
    parser.add_argument('--download', action='store_true', help='Download fresh data')
    parser.add_argument('--validate', action='store_true', help='Walk-forward validation')
    
    args = parser.parse_args()
    
    if args.live:
        run_live()
    elif args.backtest:
        run_backtest()
    elif args.download:
        run_download()
    elif args.validate:
        run_validate()
    else:
        # Default: show help
        parser.print_help()
        print("\nOr start dashboard with: python run.py --live")


if __name__ == "__main__":
    main()
