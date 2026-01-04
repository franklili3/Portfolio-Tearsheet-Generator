"""
Sample stock data for testing when yfinance API is rate limited.
This generates synthetic stock price data for AAPL and MSFT.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sample_data(symbols, start_date, end_date):
    """
    Generate sample stock data for testing when API is rate limited.

    Parameters:
    - symbols: list of stock symbols
    - start_date: start date string (YYYY-MM-DD)
    - end_date: end date string (YYYY-MM-DD)

    Returns:
    - MultiIndex DataFrame with stock data
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    # Create date range (trading days only, excluding weekends)
    dates = pd.date_range(start, end, freq='B')  # Business days

    all_data = []

    for symbol in symbols:
        # Generate synthetic price data
        np.random.seed(hash(symbol) % 2**32)  # Consistent random data per symbol

        # Starting prices (approximate historical values)
        if symbol == 'AAPL':
            base_price = 70  # Approximate price in 2020
            volatility = 0.02
        elif symbol == 'MSFT':
            base_price = 170
            volatility = 0.018
        else:
            base_price = 100
            volatility = 0.02

        # Generate price path using geometric Brownian motion
        n_days = len(dates)
        returns = np.random.normal(0.0005, volatility, n_days)  # Daily returns
        prices = base_price * (1 + returns).cumprod()

        # Create OHLCV data
        df = pd.DataFrame({
            'Open': prices * (1 + np.random.uniform(-0.01, 0.01, n_days)),
            'High': prices * (1 + np.random.uniform(0, 0.02, n_days)),
            'Low': prices * (1 + np.random.uniform(-0.02, 0, n_days)),
            'Close': prices,
            'Adj Close': prices,  # Simplified: same as close
            'Volume': np.random.randint(10000000, 100000000, n_days)
        }, index=dates)

        df['symbol'] = symbol
        all_data.append(df)

    # Combine with MultiIndex
    result = pd.concat(all_data, keys=symbols, names=['Ticker', 'Date'])
    result.index.names = ['Ticker', 'Date']

    return result

if __name__ == '__main__':
    # Test the function
    symbols = ['AAPL', 'MSFT']
    start_date = '2020-04-29'
    end_date = datetime.now().strftime('%Y-%m-%d')

    data = generate_sample_data(symbols, start_date, end_date)
    print("Sample data generated successfully!")
    print(f"\nData shape: {data.shape}")
    print(f"\nFirst few rows:\n{data.head()}")
    print(f"\nLast few rows:\n{data.tail()}")
