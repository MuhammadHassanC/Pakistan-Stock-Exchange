"""
Generate realistic synthetic historical stock data for all PSX stocks.
This creates CSV files in the data/ folder so the app works offline.
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

# Seed for reproducibility
np.random.seed(42)

# PSX stocks with realistic base prices (in PKR)
STOCKS = {
    "OGDC.KA":   {"name": "Oil & Gas Development Company", "base_price": 135, "volatility": 0.018},
    "PPL.KA":    {"name": "Pakistan Petroleum Limited",    "base_price": 95,  "volatility": 0.020},
    "PSO.KA":    {"name": "Pakistan State Oil",            "base_price": 210, "volatility": 0.022},
    "ENGRO.KA":  {"name": "Engro Corporation",             "base_price": 310, "volatility": 0.017},
    "HBL.KA":    {"name": "Habib Bank Limited",            "base_price": 145, "volatility": 0.019},
    "UBL.KA":    {"name": "United Bank Limited",           "base_price": 175, "volatility": 0.018},
    "MCB.KA":    {"name": "MCB Bank Limited",              "base_price": 195, "volatility": 0.016},
    "NBP.KA":    {"name": "National Bank of Pakistan",     "base_price": 42,  "volatility": 0.025},
    "LUCK.KA":   {"name": "Lucky Cement",                  "base_price": 780, "volatility": 0.015},
    "DGKC.KA":   {"name": "DG Khan Cement",                "base_price": 72,  "volatility": 0.023},
    "FCCL.KA":   {"name": "Fauji Cement Company",          "base_price": 28,  "volatility": 0.024},
    "HUBC.KA":   {"name": "Hub Power Company",             "base_price": 115, "volatility": 0.017},
    "KEL.KA":    {"name": "K-Electric Limited",            "base_price": 5.5, "volatility": 0.030},
    "FFC.KA":    {"name": "Fauji Fertilizer Company",      "base_price": 125, "volatility": 0.016},
    "EFERT.KA":  {"name": "Engro Fertilizers",             "base_price": 85,  "volatility": 0.019},
    "NESTLE.KA": {"name": "Nestle Pakistan",               "base_price": 6800,"volatility": 0.012},
    "COLG.KA":   {"name": "Colgate Palmolive Pakistan",    "base_price": 2400,"volatility": 0.014},
    "MEBL.KA":   {"name": "Meezan Bank Limited",           "base_price": 230, "volatility": 0.018},
    "BAHL.KA":   {"name": "Bank Al-Habib Limited",         "base_price": 88,  "volatility": 0.017},
    "SYS.KA":    {"name": "Systems Limited",               "base_price": 480, "volatility": 0.025},
    "TRG.KA":    {"name": "TRG Pakistan",                  "base_price": 135, "volatility": 0.028},
    "MARI.KA":   {"name": "Mari Petroleum",                "base_price": 1750,"volatility": 0.016},
    "POL.KA":    {"name": "Pakistan Oilfields",            "base_price": 510, "volatility": 0.017},
    "SEARL.KA":  {"name": "Searle Company",                "base_price": 62,  "volatility": 0.021},
    "ATRL.KA":   {"name": "Attock Refinery",               "base_price": 280, "volatility": 0.020},
}


def generate_stock_data(base_price, volatility, days=600, trend=0.0003):
    """
    Generate realistic OHLCV stock data using geometric Brownian motion.

    Args:
        base_price: Starting price
        volatility: Daily volatility (std dev of returns)
        days: Number of trading days to generate
        trend: Daily drift/trend factor
    """
    # Start date: ~2.5 years ago
    start_date = datetime(2024, 1, 2)
    
    # Generate trading days (skip weekends)
    dates = []
    current = start_date
    while len(dates) < days:
        if current.weekday() < 5:  # Mon-Fri
            dates.append(current)
        current += timedelta(days=1)
    
    # Generate returns using geometric Brownian motion
    # Add some realistic patterns: momentum, mean reversion, occasional spikes
    returns = np.random.normal(trend, volatility, days)
    
    # Add momentum effect (autocorrelation)
    for i in range(1, len(returns)):
        returns[i] += returns[i-1] * 0.1
    
    # Add occasional larger moves (earnings, news events)
    spike_indices = np.random.choice(days, size=int(days * 0.03), replace=False)
    returns[spike_indices] *= np.random.choice([2.5, -2.0, 3.0, -2.5], size=len(spike_indices))
    
    # Generate close prices
    close_prices = [base_price]
    for r in returns[1:]:
        new_price = close_prices[-1] * (1 + r)
        new_price = max(new_price, base_price * 0.3)  # Floor at 30% of base
        close_prices.append(new_price)
    close_prices = np.array(close_prices)
    
    # Generate OHLV from close
    daily_range = volatility * 0.8
    high_prices = close_prices * (1 + np.abs(np.random.normal(0, daily_range, days)))
    low_prices = close_prices * (1 - np.abs(np.random.normal(0, daily_range, days)))
    
    # Open is close of previous day with small gap
    open_prices = np.roll(close_prices, 1) * (1 + np.random.normal(0, volatility * 0.3, days))
    open_prices[0] = base_price
    
    # Ensure High >= max(Open, Close) and Low <= min(Open, Close)
    high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))
    low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))
    
    # Volume (realistic pattern with higher volume on bigger moves)
    base_volume = base_price * 10000
    volume = np.abs(np.random.normal(base_volume, base_volume * 0.4, days))
    volume *= (1 + np.abs(returns) * 10)  # Higher volume on big moves
    volume = volume.astype(int)
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': np.round(open_prices, 2),
        'High': np.round(high_prices, 2),
        'Low': np.round(low_prices, 2),
        'Close': np.round(close_prices, 2),
        'Volume': volume
    })
    
    return df


def main():
    """Generate and save data for all PSX stocks."""
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    print("=" * 60)
    print("  Generating PSX Stock Data")
    print("=" * 60)
    
    for ticker, info in STOCKS.items():
        # Randomize trend slightly per stock
        trend = np.random.uniform(-0.0002, 0.0006)
        days = np.random.randint(550, 620)
        
        df = generate_stock_data(
            base_price=info["base_price"],
            volatility=info["volatility"],
            days=days,
            trend=trend
        )
        
        filename = ticker.replace(".", "_") + ".csv"
        filepath = os.path.join(data_dir, filename)
        df.to_csv(filepath, index=False)
        
        latest = df['Close'].iloc[-1]
        change = ((latest - info["base_price"]) / info["base_price"]) * 100
        print(f"  [OK] {info['name']:40s} | {ticker:12s} | PKR {latest:>10,.2f} | {change:+.1f}% | {len(df)} days")
    
    print("=" * 60)
    print(f"  [DONE] Generated data for {len(STOCKS)} stocks in '{data_dir}/'")
    print("=" * 60)


if __name__ == "__main__":
    main()
