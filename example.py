# %% [markdown]
# <a href="https://colab.research.google.com/github/ldt9/Portfolio-Tearsheet-Return-Generator/blob/main/Portfolio_Tear_Sheet_Generator_with_Sector_Performace.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# # Portfolio-Tearsheet-Return-Generator
# ## Overview
# This program creates a brief, two-page document for an investor to summarize the historical outcome of the investor's portfolio transactions over the selected time period.
# 
# ## Goals of this Project
# 1. Allow investors to input their tranaction data via a .csv file
# 2. Handle time-series transactions based on First-In-First-Out (FIFO) Logic
# 3. Utilize the QuantStats library to display portfolio statistics against a benchmark
# 4. Print the statistics output to a .pdf for simpler sharing
# 
# ## Function Explanations
# 
# #### `` def create_market_cal(start, end) ``
# - Uses the pandas_market_calendars library to find all relevant trading days within a specified timeframe
# - Automatically filters out non-trading days based on the market
# - Sets NYSE as the calendar, and then standardizes the timestamps to make them easy to join on later
# 
# #### `` def read_csv(folder, csv_name) ``
# - Takes a folder name and filename for where the .csv if located and creates a dataframe out of it
# 
# #### `` def get_data(stocks, start, end) ``
# - Takes an array of stock tickers with a start and end date and grabs the data using the YFinance library
# 
# #### `` def position_adjust(daily_positions, sale) ``
# - Create an empty dataframe called stocks_with_sales to which we'll add adjusted positions, and another dataframe holding all of the transactions labeled as buys
# - Here we calculate the realized gain of each sell order
# 
# #### `` def portfolio_start_balance(portfolio, start_date) ``
# - Add to the adjusted positions dataframe, the positions that never had sales, sales that occur in the future, and any zeroed out rows to create a record of your active holdings as of the start date
# 
# #### `` def fifo(daily_positions, sales, date) ``
# - Flters sales to find any that have occurred on the current date and creates a dataframe of positions not affected by sales
# - Then use `` position_adjust `` to zero-out any positions with active sales and append the positions with no changes, leaving you with an accurate daily snapshot of your porfolio positions
# - Any rows with zero 'Qty' are left for realized gain performance calculations
# 
# #### `` def time_fill(portfolio, market_cal) ``
# - Provide our dataframe of active positions, find the sales, and zero-out sales against buy positions.
# - Loop through using our market_cal list with valid trading days
# - Filter to positions that have occurred before or at the current date and make sure there are only buys.
# - Add a Date Snapshot column with the current date in the market_cal loop, then append it to our per_day_balance list
# - Before adjusting a position, we make sure to save all future transactions to another dataframe and append them later
# 
# #### `` def modified_cost_per_share(portfolio, adj_close, start_date) ``
# - Merges provided dataframes and calculates daily adjusted cost
# 
# #### `` def portfolio_end_of_year_stats(portfolio, adj_close_end) ``
# - Finds end date closing prices for each ticker
# 
# #### `` def portfolio_start_of_year_stats(portfolio, adj_close_start) ``
# - Finds start date closing prices for each ticker
# 
# #### `` def calc_returns(portfolio) ``
# - Applies a bunch of calculations against the data we’ve been modifying, and returns a final dataframe
# 
# #### `` def per_day_portfolio_calcs(per_day_holdings, daily_adj_close, stocks_start) ``
# - Runs `` modified_cost_per_share ``, `` portfolio_end_of_year_stats ``, ``  portfolio_start_of_year_stats ``, `` calc_returns `` and returns a daily snapshot of the portfolio
# 
# #### `` def format_returns(pdpc, metric_to_group1, metric_to_group2, div, total_cash) ``
# - Formats the per day portfolio calculations for use with the QuantStats library by adding up the daily realized, unrealized gains, dividends and cash
# 
# #### `` def generate_report(returns, folder, rf=0.) ``
# - Utilizes the QuantStats library to create a .html and .pdf of the tearsheet returns as well as display the output to the consol with quantitative statistics
# 
# ## How to use this project in Google Colab
# 1. Connect to a Runtime
# 2. Press `` Ctrl + F9 ``
# 
# ## Example Quick Start Main
# ``` Python
# # Mount Google Drive
# from google.colab import drive
# drive.mount('/content/drive')
# 
# # Gather Data and Generate Report
# generate_report('Tearsheet Generator', 'TransactionHistory2020-2022withCash', '2020-04-30')
# ```
# 
# ## Example Input .csv
# | Symbol        | Qty           | Type          | Open date      | Adj cost per share | Adj cost      |
# | ------------- | ------------- | ------------- | ---------------| ------------------ | ------------- |
# | AAPL          | 10            | Buy           | 4/29/2020      | 71                 | 710           |
# | MSFT          | 10            | Buy           | 5/1/2020       | 175                | 1750          |
# | AAPL          | 5             | Sell.FIFO     | 5/15/2020      | 77                 | 385           |
# |...|...|...|...|...|...|
# 
# ## Example Output
# 
# ### HTML
# ![quantstats-tearsheet](https://user-images.githubusercontent.com/84938803/210924445-0e251786-f38b-46d3-a78d-2d2b9e6539b5.jpg)
# 
# ### PDF
# [Tearsheet_2023-01-06.pdf](https://github.com/ldt9/Portfolio-Tearsheet-Return-Generator/files/10357085/Tearsheet_2023-01-06.pdf)
# 
# ## Libraries Used
# - [YFinance](https://github.com/ranaroussi/yfinance)
# - [QuantStats](https://github.com/ranaroussi/quantstats)
# - [PdfKit](https://github.com/JazzCore/python-pdfkit)
# - [Pandas](https://github.com/pandas-dev/pandas)
# - [Pandas Market Calendars](https://github.com/rsheftel/pandas_market_calendars)
# - [MatPlotLib](https://github.com/matplotlib/matplotlib)
# 
# ## References
# - https://towardsdatascience.com/modeling-your-stock-portfolio-performance-with-python-fbba4ef2ef11
# - https://github.com/mattygyo/stock_portfolio_analysis/blob/master/portfolio_analysis.py
# - https://github.com/ranaroussi/quantstats

# %% [markdown]
# # ***Install and Import Libraries***

# %%
#%pip install yfinance==0.1.74 # For getting Price History of Assets
#%pip install quantstats --upgrade --no-cache-dir # For calculating statistics based on Portfolio Value, SRC: https://github.com/ranaroussi/quantstats
# !pip install pdfkit # For Converting an HTML file to a PdF
# !wget https://github.com/wkhtmltopdf/packaging/releases/download/0.12.6-1/wkhtmltox_0.12.6-1.bionic_amd64.deb
# !cp wkhtmltox_0.12.6-1.bionic_amd64.deb /usr/bin
# !sudo apt install /usr/bin/wkhtmltox_0.12.6-1.bionic_amd64.deb
#%pip install pandas
#%pip install pandas_market_calendars==4.1.4
#%pip install seaborn

# %%
# from datetime import date, datetime, timedelta
import datetime
import sys
import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
import pandas_market_calendars as mcal
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp
from plotly.offline import plot
from plotly.subplots import make_subplots
import itertools
import matplotlib.dates as mpl_dates
import yfinance as yf
import quantstats as qs
qs.extend_pandas()
# import pdfkit
from pathlib import Path

# %% [markdown]
# # ***Mount Drive to retrieve TXN Data***

# %%
#from google.colab import drive
#drive.mount('/content/drive')

# %% [markdown]
# # ***Market Calendar***

# %%
def create_market_cal(start, end):
  """
  Create a market calendar with trading days.
  Falls back to pandas business_days if pandas_market_calendars has compatibility issues.
  """
  try:
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start, end)
    market_cal = mcal.date_range(schedule, frequency='1D')
    market_cal = market_cal.tz_localize(None)
    market_cal = [i.replace(hour=0) for i in market_cal]
    return market_cal
  except (TypeError, AttributeError) as e:
    print(f"Warning: pandas_market_calendars error ({e}). Using simple business day calendar.")
    print("This may include holidays that are actual market closures.\n")
    # Fallback: use pandas business days (excludes weekends but may include some holidays)
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    market_cal = pd.bdate_range(start_dt, end_dt)  # Business days
    market_cal = [i.replace(hour=0) for i in market_cal]
    return market_cal

# %% [markdown]
# # ***Downloading YFinance Data***

# %%

def get_data(stocks, start, end, use_sample_data=False):
  import time

  # If explicitly requested to use sample data or all downloads fail
  if use_sample_data:
    print("Using sample data for testing...")
    from sample_stock_data import generate_sample_data
    return generate_sample_data(stocks, start, end)

  def data(ticker):
    print(f'Getting data for {ticker}...')
    # Ensure dates are in correct format for yfinance
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end) + datetime.timedelta(days=1)

    # Add retry logic for rate limiting
    max_retries = 2  # Reduced retries
    for attempt in range(max_retries):
      try:
        df = yf.download(ticker, start=start_dt.strftime('%Y-%m-%d'), end=end_dt.strftime('%Y-%m-%d'), progress=False)
        if df.empty:
          print(f'Warning: No data found for {ticker}')
          return None
        df['symbol'] = ticker
        df.index = pd.to_datetime(df.index)
        return df
      except Exception as e:
        if attempt < max_retries - 1:
          wait_time = 10
          print(f'Rate limited. Waiting {wait_time} seconds before retry...')
          time.sleep(wait_time)
        else:
          print(f'Error downloading {ticker}: {e}')
          return None

  datas = []
  for i, stock in enumerate(stocks):
    if i > 0:
      # Add delay between requests to avoid rate limiting
      print(f"Waiting 5 seconds to avoid rate limiting...")
      time.sleep(5)
    result = data(stock)
    if result is not None:
      datas.append(result)

  # If all downloads failed, offer to use sample data
  if not datas:
    print("\n" + "="*60)
    print("WARNING: All data downloads failed due to rate limiting.")
    print("="*60)
    print("\nOptions:")
    print("1. Use sample/synthetic data for testing (recommended for development)")
    print("2. Wait 1-2 hours for API rate limit to reset")
    print("3. Run the script again later")
    print("\nUsing sample data for now...\n")

    from sample_stock_data import generate_sample_data
    return generate_sample_data(stocks, start, end)

  return pd.concat(datas, keys=stocks[:len(datas)], names=['Ticker', 'Date'], sort=True)


# %% [markdown]
# # ***Read from .CSV***

# %%
def read_csv(folder, csv_name):
  portfolio_df = pd.read_csv(f'{folder}/{csv_name}.csv')
  portfolio_df['Open date'] = pd.to_datetime(portfolio_df['Open date'])
  return portfolio_df

# %% [markdown]
# # ***Portfolio Position Initialization***
# 
# We need to figure out how many shares we actively held during the start date specified. To do that, we’re going to create two functions, portfolio_start_balance and position_adjust.

# %%
def position_adjust(daily_positions, sale):
  stocks_with_sales = pd.DataFrame(columns=["Closed Stock Gain / (Loss)"])
  buys_before_start = daily_positions[daily_positions['Type'] == 'Buy'].sort_values(by='Open date')
  for position in buys_before_start[buys_before_start['Symbol'] == sale[1]['Symbol']].iterrows():
      sale[1]['Adj cost'] = pd.to_numeric(sale[1]["Adj cost"], errors='coerce')
      sale[1]['Adj cost per share'] = pd.to_numeric(sale[1]["Adj cost per share"], errors='coerce')
      position[1]['Adj cost per share'] = pd.to_numeric(position[1]["Adj cost per share"], errors='coerce')
      sale[1]['Qty'] = pd.to_numeric(sale[1]["Qty"], errors='coerce')
      if (position[1]['Qty'] <= sale[1]['Qty']) & (sale[1]['Qty'] > 0):
          position[1]["Closed Stock Gain / (Loss)"] += (sale[1]['Adj cost per share'] - position[1]['Adj cost per share']) * position[1]['Qty']
          sale[1]['Qty'] -= position[1]['Qty']
          position[1]['Qty'] = 0
      elif (position[1]['Qty'] > sale[1]['Qty']) & (sale[1]['Qty'] > 0):
          position[1]["Closed Stock Gain / (Loss)"] += (sale[1]['Adj cost per share'] - position[1]['Adj cost per share']) * sale[1]['Qty']
          position[1]['Qty'] -= sale[1]['Qty']
          sale[1]['Qty'] -= sale[1]['Qty']
      stocks_with_sales = pd.concat([stocks_with_sales, position[1].to_frame().T], ignore_index=True)
  return stocks_with_sales

def portfolio_start_balance(portfolio, start_date):
  positions_before_start = portfolio[portfolio['Open date'] <= start_date]
  future_positions = portfolio[portfolio['Open date'] >= start_date]
  sales = positions_before_start[positions_before_start['Type'] == 'Sell.FIFO'].groupby(['Symbol'])['Qty'].sum()
  sales = sales.reset_index()
  positions_no_change = positions_before_start[~positions_before_start['Symbol'].isin(sales['Symbol'].unique())]
  adj_positions_list = []
  for sale in sales.iterrows():
      adj_positions = position_adjust(positions_before_start, sale)
      if not adj_positions.empty:
        adj_positions_list.append(adj_positions)
  adj_positions_list.append(positions_no_change)
  adj_positions_list.append(future_positions)
  adj_positions_df = pd.concat(adj_positions_list, ignore_index=True)
  adj_positions_df = adj_positions_df[adj_positions_df['Qty'] > 0]
  return adj_positions_df

# %% [markdown]
# # ***FIFO - Daily Performace Snapshots***

# %%
def fifo(daily_positions, sales, date):
  sales = sales[sales['Open date'] == date]
  daily_positions = daily_positions[daily_positions['Open date'] <= date]
  positions_no_change = daily_positions[~daily_positions['Symbol'].isin(sales['Symbol'].unique())]
  adj_positions_list = []
  for sale in sales.iterrows():
      adj = position_adjust(daily_positions, sale)
      if not adj.empty:
        adj_positions_list.append(adj)
  adj_positions_list.append(positions_no_change)
  adj_positions = pd.concat(adj_positions_list, ignore_index=True) if adj_positions_list else pd.DataFrame()
  return adj_positions

def time_fill(portfolio, market_cal, stocks_end):
  portfolio['Closed Stock Gain / (Loss)'] = 0
  sales = portfolio[portfolio['Type'] == 'Sell.FIFO'].groupby(['Symbol','Open date', 'Adj cost', 'Adj cost per share'])['Qty'].sum()
  sales = sales.reset_index()
  sales['Open date'] = (sales['Open date'] + pd.DateOffset(days=1)).apply(lambda date: min(market_cal, key=lambda x: abs(x - date)))
  per_day_balance = []
  for date in market_cal:
      if (sales['Open date'] == date).any():
          future_txns = portfolio[(portfolio['Open date'] > date) & (portfolio['Open date'] <= stocks_end)]
          portfolio = fifo(portfolio, sales, date)
          portfolio = pd.concat([portfolio, future_txns], ignore_index=True)
      daily_positions = portfolio[portfolio['Open date'] <= date]
      daily_positions = daily_positions[daily_positions['Type'] == 'Buy']
      daily_positions['Date Snapshot'] = date
      per_day_balance.append(daily_positions)
  return per_day_balance


# %% [markdown]
# # ***Calculating Portfolio Return***
# 
# Now that we have an accurate by-day ledger of our active holdings, we can go ahead and create the final calculations needed to generate graphs!

# %%
def modified_cost_per_share(portfolio, adj_close, start_date):
  df = pd.merge(portfolio, adj_close, left_on=['Date Snapshot', 'Symbol'], right_on=['Date', 'Ticker'], how='left')
  df.rename(columns={'Adj Close': 'Symbol Adj Close'}, inplace=True)
  df['Adj cost daily'] = df['Symbol Adj Close'] * df['Qty']
  df = df.drop(['Ticker', 'Date'], axis=1)
  return df

def portfolio_end_of_year_stats(portfolio, adj_close_end):
  adj_close_end = adj_close_end[adj_close_end['Date'] == adj_close_end['Date'].max()]
  portfolio_end_data = pd.merge(portfolio, adj_close_end, left_on='Symbol', right_on='Ticker')
  portfolio_end_data.rename(columns={'Adj Close': 'Ticker End Date Close'}, inplace=True)
  portfolio_end_data = portfolio_end_data.drop(['Ticker', 'Date'], axis=1)
  return portfolio_end_data

def portfolio_start_of_year_stats(portfolio, adj_close_start):
  adj_close_start = adj_close_start[adj_close_start['Date'] == adj_close_start['Date'].min()]
  portfolio_start = pd.merge(portfolio, adj_close_start[['Ticker', 'Adj Close', 'Date']], left_on='Symbol', right_on='Ticker')
  portfolio_start.rename(columns={'Adj Close': 'Ticker Start Date Close'}, inplace=True)
  portfolio_start['Adj cost per share'] = np.where(portfolio_start['Open date'] <= portfolio_start['Date'], portfolio_start['Ticker Start Date Close'], portfolio_start['Adj cost per share'])
  portfolio_start["Adj cost per share"] = pd.to_numeric(portfolio_start["Adj cost per share"], errors='coerce')
  portfolio_start['Adj cost'] = portfolio_start['Adj cost per share'] * portfolio_start['Qty']
  portfolio_start = portfolio_start.drop(['Ticker', 'Date'], axis=1)
  return portfolio_start

def calc_returns(portfolio):
  portfolio['Ticker Daily Return'] = portfolio.groupby('Symbol')['Symbol Adj Close'].pct_change()
  portfolio['Ticker Return'] = portfolio['Symbol Adj Close'] / portfolio['Adj cost per share'] - 1
  portfolio['Ticker Share Value'] = portfolio['Qty'] * portfolio['Symbol Adj Close']
  portfolio['Open Stock Gain / (Loss)'] = portfolio['Ticker Share Value'] - portfolio['Adj cost']
  portfolio = portfolio.dropna(axis=1, how='all') # Drops some columns that get added in by mistake
  return portfolio

def per_day_portfolio_calcs(per_day_holdings, daily_adj_close, stocks_start):
  df = pd.concat(per_day_holdings, sort=True)
  mcps = modified_cost_per_share(df, daily_adj_close, stocks_start)
  pes = portfolio_end_of_year_stats(mcps, daily_adj_close)
  pss = portfolio_start_of_year_stats(pes, daily_adj_close)
  returns = calc_returns(pss)
  return returns

# ROLLING SHARE VALUE METHOD
def format_returns(pdpc):
  # Sum of the ticker share value on each day
  Ticker_Share_Value = pdpc.groupby(['Date Snapshot'])[['Ticker Share Value']].sum().reset_index()
  Ticker_Share_Value = pd.melt(Ticker_Share_Value, id_vars=['Date Snapshot'],
                              value_vars=['Ticker Share Value'])
  Ticker_Share_Value.set_index('Date Snapshot', inplace=True)
  Ticker_Share_Value.rename(columns={'value': 'Ticker Share Value'}, inplace=True)

  # Total Ticker Share Value Weighted Return on each day
  grouped_metrics5 = pdpc.groupby(['Date Snapshot', 'Symbol'])['Ticker Share Value'].sum().reset_index()
  grouped_metrics5 = grouped_metrics5.merge(Ticker_Share_Value, on='Date Snapshot', suffixes=('', '_total'))
  grouped_metrics5['Weight'] = grouped_metrics5['Ticker Share Value'] / grouped_metrics5['Ticker Share Value_total']

  # Calculate daily returns for each symbol
  symbol_returns = pdpc.groupby(['Date Snapshot', 'Symbol'])['Ticker Daily Return'].sum().reset_index()

  # Join the `grouped_metrics5` dataframe with the `symbol_returns` series
  grouped_metrics5 = pd.concat([grouped_metrics5, symbol_returns['Ticker Daily Return']], axis=1, join='inner')

  # Calculate ticker weighted returns
  grouped_metrics5['Ticker Weighted Return'] = grouped_metrics5['Weight'] * grouped_metrics5['Ticker Daily Return']

  # Group by date and calculate total weighted returns
  grouped_metrics5 = grouped_metrics5.groupby('Date Snapshot')['Ticker Weighted Return'].sum().reset_index()
  grouped_metrics5.rename(columns={'Ticker Weighted Return': 'Total Weighted Return'}, inplace=True)
  grouped_metrics5.set_index('Date Snapshot', inplace=True)

  # calculate daily returns in % form
  grouped_metrics5['Total Weighted Return'].fillna(0, inplace=True)
  grouped_metrics5['Total Weighted Return'] = grouped_metrics5['Total Weighted Return'].replace([np.inf, -np.inf], 0)
  grouped_metrics5['Cumulative Total Weighted Return'] = (grouped_metrics5['Total Weighted Return'].cumsum() * 100).ffill()

  # Plot Data
  # line(grouped_metrics5, 'Total Weighted Return')
  # line(grouped_metrics5, 'Cumulative Total Weighted Return')

  return grouped_metrics5['Total Weighted Return']

# STATIC COST BASIS METHOD
# def format_returns(pdpc):
#   # Sum of the ticker share value on each day
#   Ticker_Share_Value = pdpc.groupby(['Date Snapshot'])[['Adj cost']].sum().reset_index()
#   Ticker_Share_Value = pd.melt(Ticker_Share_Value, id_vars=['Date Snapshot'],
#                               value_vars=['Adj cost'])
#   Ticker_Share_Value.set_index('Date Snapshot', inplace=True)
#   Ticker_Share_Value.rename(columns={'value': 'Adj cost'}, inplace=True)

#   # Total Ticker Share Value Weighted Return on each day
#   grouped_metrics5 = pdpc.groupby(['Date Snapshot', 'Symbol'])['Adj cost'].sum().reset_index()
#   grouped_metrics5 = grouped_metrics5.merge(Ticker_Share_Value, on='Date Snapshot', suffixes=('', '_total'))
#   grouped_metrics5['Weight'] = grouped_metrics5['Adj cost'] / grouped_metrics5['Adj cost_total']

#   # Calculate daily returns for each symbol
#   symbol_returns = pdpc.groupby(['Date Snapshot', 'Symbol'])['Ticker Daily Return'].sum().reset_index()

#   # Join the `grouped_metrics5` dataframe with the `symbol_returns` series
#   grouped_metrics5 = pd.concat([grouped_metrics5, symbol_returns['Ticker Daily Return']], axis=1, join='inner')

#   # Calculate ticker weighted returns
#   grouped_metrics5['Ticker Weighted Return'] = grouped_metrics5['Weight'] * grouped_metrics5['Ticker Daily Return']

#   # Group by date and calculate total weighted returns
#   grouped_metrics5 = grouped_metrics5.groupby('Date Snapshot')['Ticker Weighted Return'].sum().reset_index()
#   grouped_metrics5.rename(columns={'Ticker Weighted Return': 'Total Weighted Return'}, inplace=True)
#   grouped_metrics5.set_index('Date Snapshot', inplace=True)

#   # calculate daily returns in % form
#   grouped_metrics5['Total Weighted Return'].fillna(0, inplace=True)
#   grouped_metrics5['Total Weighted Return'] = grouped_metrics5['Total Weighted Return'].replace([np.inf, -np.inf], 0)
#   grouped_metrics5['Cumulative Total Weighted Return'] = (grouped_metrics5['Total Weighted Return'].cumsum() * 100).ffill()

#   # plot some test lines
#   line(grouped_metrics5, 'Total Weighted Return')
#   line(grouped_metrics5, 'Cumulative Total Weighted Return')

#   return grouped_metrics5['Total Weighted Return']


# %% [markdown]
# # ***Test Plots***
# 
# These will help us see whats going on in the spreadsheet visually without generating the whole report.

# %%
def line_facets(df, val_1, output_dir=None):
    grouped_metrics = df.groupby(['Symbol','Date Snapshot'])[[val_1]].sum().reset_index()
    grouped_metrics = pd.melt(grouped_metrics, id_vars=['Symbol','Date Snapshot'],
                              value_vars=[val_1])
    fig = px.line(grouped_metrics, x="Date Snapshot", y="value",
                  color='variable', facet_col="Symbol", facet_col_wrap=10)
    if output_dir:
        fig.write_html(str(output_dir / "lineGraph.html"))
    else:
        fig.write_html("lineGraph.html")
    plot(fig)
    fig.show()

def line(df, val_1, output_dir=None):
    grouped_metrics = df.groupby(['Date Snapshot'])[[val_1]].sum().reset_index()
    grouped_metrics = pd.melt(grouped_metrics, id_vars=['Date Snapshot'],
                              value_vars=[val_1])
    fig = px.line(grouped_metrics, x="Date Snapshot", y="value",
                  color='variable')
    plot(fig)
    fig.show()

# %% [markdown]
# # ***Add the Sector Type to Each ticker in the Portfolio Dataframe***

# %%
def assign_sectors(df):
  sectors_dict = {
        'Technology' : ['AAPL','ADBE', 'LRCX', 'MSFT','NVDA', 'VGT','CRM'],
        'Financials & Real Estate' : ['BLK', 'BN', 'SIVBQ', 'VFH','BAM','RLI','VNQ'],
        'Healthcare' : ['ABBV', 'MDT', 'TMO','ELV','VHT','MRK', 'OGN', 'UNH'],
        'Consumer Discretionary' : ['LULU','TJX','VCR'],
        'Communications' : ['DIS', 'META', 'TMUS', 'GOOGL','VOX','ATVI'],
        'Industrials' : ['CAT','J','ROK','UNP','VIS'],
        'Consumer Staples' : ['WMT','VDC'],
        'Utilities' : ['AEP','NEE', 'SRE','UGI','VPU','AES','AWK'],
        'Materials' : ['VAW','SLGN'],
        'Energy' : ['VDE','EOG']
        # 'Real Estate' : ['VNQ']
        # 'Cash' : ['VMFXX']
    }

  sectors = []
  for ticker in df['Symbol']:
      found = False
      for sector, stocks in sectors_dict.items():
          if ticker in stocks:
              sectors.append(sector)
              found = True
              break

      if not found:
          sectors.append('None')

  df['Sector'] = sectors
  return df

def sector_perf(pdpc, output_dir):

  # Get SPY for Benchmarking
  benchmarks = ['SPY']
  spy = get_data(benchmarks, pdpc['Date Snapshot'].min().strftime("%Y-%m-%d"), pdpc['Date Snapshot'].max().strftime("%Y-%m-%d"))
  spy = spy.droplevel('Ticker')
  spy['pct_change'] = spy['Close'].pct_change()
  spy['Cumulative Total Return'] = spy['pct_change'].cumsum() * 100

  # Empty DF for plotting individual symbols by sector based on sector weighting
  symbols_combined = pd.DataFrame()

  fig_sector_performance = go.Figure()

  for i, sector in enumerate(pdpc['Sector'].unique()):

    # Filter the pdpc for each sector
    sector_df = pdpc[pdpc['Sector'] == sector]

    # Sum of the ticker share value on each day
    Ticker_Share_Value = sector_df.groupby(['Date Snapshot'])[['Ticker Share Value']].sum().reset_index()
    Ticker_Share_Value = pd.melt(Ticker_Share_Value, id_vars=['Date Snapshot'],
                                value_vars=['Ticker Share Value'])
    Ticker_Share_Value.set_index('Date Snapshot', inplace=True)
    Ticker_Share_Value.rename(columns={'value': 'Ticker Share Value'}, inplace=True)

    # Total Ticker Share Value Weighted Return on each day
    grouped_metrics5 = sector_df.groupby(['Date Snapshot', 'Symbol'])['Ticker Share Value'].sum().reset_index()
    grouped_metrics5 = grouped_metrics5.merge(Ticker_Share_Value, on='Date Snapshot', suffixes=('', '_total'))
    grouped_metrics5['Weight'] = grouped_metrics5['Ticker Share Value'] / grouped_metrics5['Ticker Share Value_total']

    # Calculate daily returns for each symbol
    symbol_returns = sector_df.groupby(['Date Snapshot', 'Symbol'])['Ticker Daily Return'].sum().reset_index()

    # Join the `grouped_metrics5` dataframe with the `symbol_returns` series
    grouped_metrics5 = pd.concat([grouped_metrics5, symbol_returns['Ticker Daily Return']], axis=1, join='inner')

    # Calculate ticker weighted returns
    grouped_metrics5['Ticker Weighted Return'] = grouped_metrics5['Weight'] * grouped_metrics5['Ticker Daily Return']
    symbols_combined = pd.concat([symbols_combined, grouped_metrics5], ignore_index=True)

    # Group by date and calculate total weighted returns
    grouped_metrics5 = grouped_metrics5.groupby('Date Snapshot')['Ticker Weighted Return'].sum().reset_index()
    grouped_metrics5.rename(columns={'Ticker Weighted Return': 'Total Weighted Return'}, inplace=True)
    grouped_metrics5.set_index('Date Snapshot', inplace=True)

    # calculate daily returns in % form
    grouped_metrics5['Total Weighted Return'].fillna(0, inplace=True)
    grouped_metrics5['Total Weighted Return'] = grouped_metrics5['Total Weighted Return'].replace([np.inf, -np.inf], 0)
    grouped_metrics5['Cumulative Total Weighted Return'] = (grouped_metrics5['Total Weighted Return'].cumsum() * 100).ffill()

    colors = (['red', 'green', 'blue', 'black', 'orange', 'darkviolet', 'slategrey', 'fuchsia', 'maroon', 'saddlebrown', 'greenyellow', 'cyan'])
    # Add plot to figure
    # plt.plot(grouped_metrics5.index, grouped_metrics5['Cumulative Total Weighted Return'], label=f"{sector}: {grouped_metrics5['Cumulative Total Weighted Return'].iloc[-2]:.2f}%", color=colors[i])
    fig_sector_performance.add_trace(
        go.Scatter(
            x=grouped_metrics5.index,
            y=grouped_metrics5['Cumulative Total Weighted Return'],
            mode='lines',
            name=f"{sector}: {grouped_metrics5['Cumulative Total Weighted Return'].iloc[-2]:.2f}%",
            line=dict(color=colors[i])
        )
    )


  # Add Benchmark
  # plt.plot(spy.index, spy['Cumulative Total Return'], label=f"SPY: {spy['Cumulative Total Return'].iloc[-2]:.2f}%", color='purple')
  fig_sector_performance.add_trace(
      go.Scatter(
          x=spy.index,
          y=spy['Cumulative Total Return'],
          mode='lines',
          name=f"SPY: {spy['Cumulative Total Return'].iloc[-2]:.2f}%",
          line=dict(color='purple')
      )
  )

  today = datetime.datetime.today().strftime("%Y-%m-%d")

  # plt.title(f'Sector Performance as of {today}')
  # plt.xlabel('Date Snapshot')
  # plt.ylabel('Returns (%)')
  # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

  # fig = plt.gcf()
  # fig.set_size_inches(18, 10)

  # plt.show()
  fig_sector_performance.update_layout(
        title=f'Sector Performance as of {today}',
        xaxis_title='Date Snapshot',
        yaxis_title='Returns (%)',
        legend=dict(x=1, y=0.5),
        autosize=False,
        width=1000*2,
        height=600*2,
    )

  fig_sector_performance.write_html(str(output_dir / "fig_sector_performance.html"))
  fig_sector_performance.show()

  for sector in pdpc['Sector'].unique():
    # Filter the pdpc for each sector
    sector_df = pdpc[pdpc['Sector'] == sector]

    # Get a list of unique symbols within the sector
    symbols = sector_df['Symbol'].unique()

    # Create a new figure for each sector
    # plt.figure()
    fig_individual_sector = go.Figure()

    for i, symbol in enumerate(symbols):
      # Filter the sector_df for each symbol in the sector
      symbol_df = symbols_combined[symbols_combined['Symbol'] == symbol]
      # symbol_df = sector_df[sector_df['Symbol'] == symbol]

      # Returns based on individual holdings (weighting of 1 for each)
      # Sum of the ticker share value on each day
      Ticker_Share_Value = symbol_df.groupby(['Date Snapshot'])[['Ticker Share Value']].sum().reset_index()
      Ticker_Share_Value = pd.melt(Ticker_Share_Value, id_vars=['Date Snapshot'],
                                  value_vars=['Ticker Share Value'])
      Ticker_Share_Value.set_index('Date Snapshot', inplace=True)
      Ticker_Share_Value.rename(columns={'value': 'Ticker Share Value'}, inplace=True)

      # Total Ticker Share Value Weighted Return on each day
      grouped_metrics5 = symbol_df.groupby(['Date Snapshot', 'Symbol'])['Ticker Share Value'].sum().reset_index()
      grouped_metrics5 = grouped_metrics5.merge(Ticker_Share_Value, on='Date Snapshot', suffixes=('', '_total'))
      grouped_metrics5['Weight'] = grouped_metrics5['Ticker Share Value'] / grouped_metrics5['Ticker Share Value_total']

      # Calculate daily returns for each symbol
      symbol_returns = symbol_df.groupby(['Date Snapshot', 'Symbol'])['Ticker Daily Return'].sum().reset_index()

      # Join the `grouped_metrics5` dataframe with the `symbol_returns` series
      grouped_metrics5 = pd.concat([grouped_metrics5, symbol_returns['Ticker Daily Return']], axis=1, join='inner')

      # Calculate ticker weighted returns
      grouped_metrics5['Ticker Weighted Return'] = grouped_metrics5['Weight'] * grouped_metrics5['Ticker Daily Return']

      # Returns based on weighting in Sector
      grouped_metrics5 = symbol_df

      # Group by date and calculate total weighted returns
      # grouped_metrics5 = grouped_metrics5.groupby('Date Snapshot')['Ticker Weighted Return'].sum().reset_index()
      grouped_metrics5.rename(columns={'Ticker Weighted Return': 'Total Weighted Return'}, inplace=True)
      grouped_metrics5.set_index('Date Snapshot', inplace=True)

      # calculate daily returns in % form
      grouped_metrics5['Total Weighted Return'].fillna(0, inplace=True)
      grouped_metrics5['Total Weighted Return'] = grouped_metrics5['Total Weighted Return'].replace([np.inf, -np.inf], 0)
      grouped_metrics5['Cumulative Total Weighted Return'] = (grouped_metrics5['Total Weighted Return'].cumsum() * 100).ffill()

      colors = (['red', 'green', 'blue', 'black', 'orange', 'darkviolet', 'slategrey', 'fuchsia', 'maroon', 'saddlebrown', 'greenyellow', 'cyan'])
      # Add plot to figure
      # plt.plot(grouped_metrics5.index, grouped_metrics5['Cumulative Total Weighted Return'], label=f"{symbol}: {grouped_metrics5['Cumulative Total Weighted Return'].iloc[-2]:.2f}%", color=colors[i])
      fig_individual_sector.add_trace(
                      go.Scatter(
                          x=grouped_metrics5.index,
                          y=grouped_metrics5['Cumulative Total Weighted Return'],
                          mode='lines',
                          name=f"{symbol}: {grouped_metrics5['Cumulative Total Weighted Return'].iloc[-2]:.2f}%",
                          line=dict(color=colors[i])
                      )
                  )

    # Add Benchmark
    # plt.plot(spy.index, spy['Cumulative Total Return'], label=f"SPY: {spy['Cumulative Total Return'].iloc[-2]:.2f}%", color='purple')
    fig_individual_sector.add_trace(
        go.Scatter(
            x=spy.index,
            y=spy['Cumulative Total Return'],
            mode='lines',
            name=f"SPY: {spy['Cumulative Total Return'].iloc[-2]:.2f}%",
            line=dict(color='purple')
        )
    )

    # Set plot title, labels, and legend for the current sector
    # plt.title(f'Sector Performance - {sector} as of {today}')
    # plt.xlabel('Date Snapshot')
    # plt.ylabel('Returns (%)')
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # fig = plt.gcf()
    # fig.set_size_inches(18, 10)

    # plt.show()

    fig_individual_sector.update_layout(
        title=f'Sector Performance - {sector} as of {today}',
        xaxis_title='Date Snapshot',
        yaxis_title='Returns (%)',
        legend=dict(x=1, y=0.5),
        autosize=False,
        width=1000*2,
        height=600*2,
    )

    fig_individual_sector.write_html(str(output_dir / f"fig_individual_sector_{sector}.html"))
    fig_individual_sector.show()

# %% [markdown]
# # ***Report Generation***
# 
# Utilize the QuantStats Library to generate a report for our Portfolio

# %%
def generate_report(folder, filename, start_date, sector='Portfolio', rf=0.02):
  print("Reading Portfolio Transactions from .CSV...")

  # Create output directory
  output_dir = Path('./output')
  output_dir.mkdir(parents=True, exist_ok=True)
  print(f"Output will be saved to: {output_dir.absolute()}")

  portfolio_df = read_csv(folder, filename)

  # create a mask to filter the dataframe to only contain buy and sell orders
  mask = (portfolio_df['Type'] == 'Buy') | (portfolio_df['Type'] == 'Sell.FIFO')

  # save dividends to a seperate df before applying the mask
  dividends = portfolio_df[portfolio_df['Type'] == 'Dividend']

  portfolio_df = portfolio_df[mask]

  # Create and Array of Unique Tickers
  symbols = portfolio_df.Symbol.unique()
  print(symbols)

  # Add Sector to each Ticker
  portfolio_df = assign_sectors(portfolio_df)

  today = datetime.datetime.today()
  stocks_end = today.strftime("%Y-%m-%d")

  daily_adj_close = get_data(symbols, start_date, stocks_end)
  daily_adj_close = daily_adj_close[['Adj Close']].reset_index()

  market_cal = create_market_cal(start_date, stocks_end)

  print('Determining Active Portfolio...')

  active_portfolio = portfolio_start_balance(portfolio_df, start_date)

  print('Calculating Daily Position Snapshots...')

  positions_per_day = time_fill(active_portfolio, market_cal, stocks_end)

  pdpc = per_day_portfolio_calcs(positions_per_day, daily_adj_close, start_date)

  # Check Daily Snapshots in .csv format
  filepath = output_dir / f'PerDayPortfolioCalculations_{datetime.date.today()}.csv'
  pdpc.to_csv(filepath)

  print("Generating Sector Performance Plot")

  sector_perf(pdpc, output_dir)

  print('Formatting Returns for QuantStats Library & Plotting Individual Sector Returns...')

  portReturns = format_returns(pdpc)

  print("Generating Report...")

  # Portfolio Metrics
  metrics = qs.reports.metrics(portReturns, 'SPY', rf=rf, display=False, mode='Full')
  metrics['Statistics'] = metrics.index
  print(metrics)
  metrics.to_csv(output_dir / f"metrics_{sector}.csv")

  rolling_beta = qs.stats.rolling_greeks(portReturns, 'SPY')
  print(rolling_beta)
  rolling_beta.to_csv(output_dir / f"rolling_beta_{sector}.csv")

  rolling_sharpe = pd.DataFrame(qs.stats.rolling_sharpe(portReturns, rf=rf))
  rolling_sharpe.rename(columns={'Total Weighted Return':'Rolling Sharpe Ratio'}, inplace=True)
  print(rolling_sharpe)
  rolling_sharpe.to_csv(output_dir / f"rolling_sharpe_{sector}.csv")

  rolling_volatility = pd.DataFrame(qs.stats.rolling_volatility(portReturns))
  rolling_volatility.rename(columns={'Total Weighted Return':'Rolling Volatility'}, inplace=True)
  print(rolling_volatility)
  rolling_volatility.to_csv(output_dir / f"rolling_volatility_{sector}.csv")
  # rolling_volatility = qs.plots.rolling_volatility(portReturns, 'SPY')

  # Extract additional chart data for CSV export
  print("Exporting additional chart data...")

  # 1. Cumulative Returns Data
  cumulative_returns = pd.DataFrame({
      'Date': portReturns.index,
      'Daily Return': portReturns.values,
      'Cumulative Return': (1 + portReturns).cumprod().values - 1
  })
  cumulative_returns.set_index('Date', inplace=True)
  cumulative_returns.to_csv(output_dir / f"cumulative_returns_{sector}.csv")
  print(f"✓ Saved cumulative_returns_{sector}.csv")

  # 2. EOY (End of Year) Returns
  try:
    # Calculate yearly returns manually
    yearly_returns = portReturns.resample('Y').apply(lambda x: (1 + x).prod() - 1)
    yearly_returns_df = pd.DataFrame({
        'Year': yearly_returns.index.year,
        'Strategy Return': yearly_returns.values * 100
    })
    yearly_returns_df.to_csv(output_dir / f"eoy_returns_{sector}.csv", index=False)
    print(f"✓ Saved eoy_returns_{sector}.csv")
  except Exception as e:
    print(f"Warning: Could not export EOY returns: {e}")

  # 3. Worst Drawdowns Data
  try:
    drawdown_series = qs.stats.to_drawdown_series(portReturns)
    # Find drawdown periods
    drawdown_df = pd.DataFrame(drawdown_series)
    drawdown_df.columns = ['Drawdown']
    drawdown_df.to_csv(output_dir / f"drawdown_series_{sector}.csv")
    print(f"✓ Saved drawdown_series_{sector}.csv")
  except Exception as e:
    print(f"Warning: Could not export drawdown details: {e}")

  # 4. Monthly Returns Heatmap Data
  try:
    monthly_returns = qs.stats.monthly_returns(portReturns, eoy=True)
    monthly_returns.to_csv(output_dir / f"monthly_returns_heatmap_data_{sector}.csv")
    print(f"✓ Saved monthly_returns_heatmap_data_{sector}.csv")
  except Exception as e:
    print(f"Warning: Could not export monthly returns heatmap data: {e}")

  # 5. Daily Returns Data
  daily_returns_df = pd.DataFrame({
      'Date': portReturns.index,
      'Daily Return': portReturns.values,
      'Cumulative Returns': (1 + portReturns).cumprod().values - 1
  })
  daily_returns_df.to_csv(output_dir / f"daily_returns_data_{sector}.csv", index=False)
  print(f"✓ Saved daily_returns_data_{sector}.csv")

  # 6. Key Statistics Summary
  try:
    key_stats = pd.Series({
        'Total Return': qs.stats.comp(portReturns),
        'CAGR': qs.stats.cagr(portReturns),
        'Sharpe Ratio': qs.stats.sharpe(portReturns, rf=rf),
        'Sortino Ratio': qs.stats.sortino(portReturns, rf=rf),
        'Max Drawdown': qs.stats.max_drawdown(portReturns),
        'Volatility (Ann.)': qs.stats.volatility(portReturns),
        'Win Rate': qs.stats.win_rate(portReturns),
        'Best Day': qs.stats.best(portReturns),
        'Worst Day': qs.stats.worst(portReturns),
        'Avg Win': qs.stats.avg_win(portReturns),
        'Avg Loss': qs.stats.avg_loss(portReturns),
        'Profit Factor': qs.stats.profit_factor(portReturns)
    })
    key_stats.to_csv(output_dir / f"key_statistics_summary_{sector}.csv")
    print(f"✓ Saved key_statistics_summary_{sector}.csv")
  except Exception as e:
    print(f"Warning: Could not export key statistics: {e}")

  # monthly_returns_heatmap = qs.stats.monthly_returns(portReturns, eoy=False)
  monthly_returns_heatmap = qs.plots.monthly_returns(portReturns, savefig=str(output_dir / f"monthly_returns_heatmap_{sector}.png"))

  # monthly_returns_distribution = qs.stats.distribution(portReturns)
  monthly_returns_distribution = qs.plots.histogram(portReturns, savefig=str(output_dir / f"monthly_returns_distribution_{sector}.png"))

  # drawdown = qs.stats.to_drawdown_series(portReturns)
  drawdown = qs.plots.drawdown(portReturns, savefig=str(output_dir / f"drawdown_{sector}.png"))

  daily_returns = qs.plots.daily_returns(portReturns, 'SPY', savefig=str(output_dir / f"daily_returns_{sector}.png"))

  # Generate additional charts from QuantStats
  print("Generating additional QuantStats charts...")

  # 1. Cumulative Returns Chart
  try:
    qs.plots.returns(portReturns, 'SPY', savefig=str(output_dir / f"cumulative_returns_chart_{sector}.png"))
    print(f"✓ Saved cumulative_returns_chart_{sector}.png")
  except Exception as e:
    print(f"Warning: Could not generate cumulative returns chart: {e}")

  # 2. Log Returns Chart
  try:
    qs.plots.log_returns(portReturns, 'SPY', savefig=str(output_dir / f"log_returns_{sector}.png"))
    print(f"✓ Saved log_returns_{sector}.png")
  except Exception as e:
    print(f"Warning: Could not generate log returns chart: {e}")

  # 3. Rolling Volatility Chart
  try:
    qs.plots.rolling_volatility(portReturns, savefig=str(output_dir / f"rolling_volatility_chart_{sector}.png"))
    print(f"✓ Saved rolling_volatility_chart_{sector}.png")
  except Exception as e:
    print(f"Warning: Could not generate rolling volatility chart: {e}")

  # 4. Rolling Sharpe Ratio Chart
  try:
    qs.plots.rolling_sharpe(portReturns, rf=rf, savefig=str(output_dir / f"rolling_sharpe_chart_{sector}.png"))
    print(f"✓ Saved rolling_sharpe_chart_{sector}.png")
  except Exception as e:
    print(f"Warning: Could not generate rolling sharpe chart: {e}")

  # 5. Rolling Sortino Ratio Chart
  try:
    qs.plots.rolling_sortino(portReturns, rf=rf, savefig=str(output_dir / f"rolling_sortino_chart_{sector}.png"))
    print(f"✓ Saved rolling_sortino_chart_{sector}.png")
  except Exception as e:
    print(f"Warning: Could not generate rolling sortino chart: {e}")

  # 6. Rolling Beta Chart
  try:
    qs.plots.rolling_beta(portReturns, 'SPY', savefig=str(output_dir / f"rolling_beta_chart_{sector}.png"))
    print(f"✓ Saved rolling_beta_chart_{sector}.png")
  except Exception as e:
    print(f"Warning: Could not generate rolling beta chart: {e}")

  # 7. Yearly Returns Chart
  try:
    qs.plots.yearly_returns(portReturns, 'SPY', savefig=str(output_dir / f"yearly_returns_{sector}.png"))
    print(f"✓ Saved yearly_returns_{sector}.png")
  except Exception as e:
    print(f"Warning: Could not generate yearly returns chart: {e}")

  # 8. Drawdown Periods Chart
  try:
    qs.plots.drawdowns_periods(portReturns, savefig=str(output_dir / f"drawdown_periods_{sector}.png"))
    print(f"✓ Saved drawdown_periods_{sector}.png")
  except Exception as e:
    print(f"Warning: Could not generate drawdown periods chart: {e}")

  # 9. Earnings Chart (if applicable)
  try:
    qs.plots.earnings(portReturns, savefig=str(output_dir / f"earnings_{sector}.png"))
    print(f"✓ Saved earnings_{sector}.png")
  except Exception as e:
    print(f"Note: Earnings chart not applicable: {e}")

  # 10. Snapshot Summary Chart
  try:
    qs.plots.snapshot(portReturns, 'SPY', title=f'{sector} Performance Snapshot',
                     savefig=str(output_dir / f"snapshot_{sector}.png"))
    print(f"✓ Saved snapshot_{sector}.png")
  except Exception as e:
    print(f"Warning: Could not generate snapshot chart: {e}")

  histogram = metrics.loc[metrics.index.isin(['MTD', '3M', '6M', 'YTD', '1Y']), ['Benchmark (SPY)', 'Strategy']]
  df = pd.DataFrame(histogram, index=['MTD', '3M', '6M', 'YTD', '1Y'])

  # Create a grouped bar plot
  sns.set_theme(style="whitegrid")
  plt.figure(figsize=(10, 6))

  ax = df.plot(kind='bar', width=0.7)
  ax.set_title('Comparison of Benchmark vs. Strategy Returns')
  ax.set_xlabel('Time Interval')
  ax.set_ylabel('Returns (%)')  # Set the y-axis label

  # Convert y-axis labels to percentages
  def percentage(x, pos):
      return f'{100 * x:.1f}%'

  ax.yaxis.set_major_formatter(FuncFormatter(percentage))
  ax.legend(title='Category')
  ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

  # Add labels at the end of each bar
  for p in ax.patches:
      ax.annotate(f'{p.get_height():.1%}',
                  (p.get_x() + p.get_width() / 2., p.get_height()),
                  ha='center', va='center', fontsize=10, color='black',
                  xytext=(0, 10), textcoords='offset points')
  plt.tight_layout()
  plt.savefig(output_dir / f"histogram_{sector}.png")
  plt.show()

  full = qs.reports.full(portReturns, benchmark='SPY', rf=rf, grayscale=False,
          figsize=(8, 5), display=True, compounded=True,
          periods_per_year=252, match_dates=False)

  doc = qs.reports.html(portReturns, benchmark='SPY', rf=rf, grayscale=False,
          title=f'{sector} Investment Returns', output=str(output_dir / f'{sector}_Tearsheet_{datetime.date.today()}.html'), compounded=True,
          periods_per_year=252, download_filename='quantstats-tearsheet.html',
          figfmt='svg', template_path=None, match_dates=False)

  # Currently broken due to pdfkit issues - WIP
  # options = {
  #     'page-size': 'B4',
  #     'margin-top': '0.10in',
  #     'margin-right': '0.0in',
  #     'margin-bottom': '0.0in',
  #     'margin-left': '0.10in'
  #     }
  # pdfkit.from_file(f'/content/drive/MyDrive/Colab Notebooks/{folder}/{sector}_Tearsheet_{datetime.date.today()}.html', f'/content/drive/MyDrive/Colab Notebooks/{folder}/{sector}_Tearsheet_{datetime.date.today()}.pdf', options=options)

  # pdfkit.from_file()

  # print("Report Generated in Colab as well as .html and .pdf")


# %% [markdown]
# # ***MAIN***

# %%
import warnings
warnings.filterwarnings("ignore")

generate_report('./', 'exchange_transactions', '2020-04-29')
'''
# %%
# Make Sector Tearsheets

# Define Sector Dictionary
sectors_dict = {
      'Technology' : ['AAPL','ADBE', 'LRCX', 'MSFT','NVDA', 'VGT','CRM'],
      'Financials & Real Estate' : ['BLK', 'BN', 'SIVBQ', 'VFH','BAM','RLI','VNQ'],
      'Healthcare' : ['ABBV', 'MDT', 'TMO','ELV','VHT','MRK', 'OGN', 'UNH'],
      'Consumer Discretionary' : ['LULU','TJX','VCR'],
      'Communications' : ['DIS', 'META', 'TMUS', 'GOOGL','VOX','ATVI'],
      'Industrials' : ['CAT','J','ROK','UNP','VIS'],
      'Consumer Staples' : ['WMT','VDC'],
      'Utilities' : ['AEP','NEE', 'SRE','UGI','VPU','AES','AWK'],
      'Materials' : ['VAW','SLGN'],
      'Energy' : ['VDE','EOG']
      # 'Real Estate' : ['VNQ']
      # 'Cash' : ['VMFXX']
  }

# Load the transaction data from the CSV file
portfolio_df = read_csv('./', 'exchange_transactions')

# Iterate over the sectors in the sectors_dict
for sector, symbols in sectors_dict.items():
    # Filter the transaction data for symbols in the current sector
    sector_data = portfolio_df[portfolio_df['Symbol'].isin(symbols)]

    # Save the filtered data as a new CSV file
    sector_data.to_csv(f'./{sector}_TransactionHistory2020-2022_withCash.csv', index=False)

# Run Generate Report for each Sector
generate_report('./', 'Technology_TransactionHistory2020-2022_withCash', '2020-04-30', 'Technology')
generate_report('./', 'Financials & Real Estate_TransactionHistory2020-2022_withCash', '2020-04-30', 'Financials & Real Estate')
generate_report('./', 'Healthcare_TransactionHistory2020-2022_withCash', '2020-04-30', 'Healthcare')
generate_report('./', 'Consumer Discretionary_TransactionHistory2020-2022_withCash', '2020-04-30', 'Consumer Discretionary')
generate_report('./', 'Communications_TransactionHistory2020-2022_withCash', '2020-04-30', 'Communications')
generate_report('./', 'Industrials_TransactionHistory2020-2022_withCash', '2020-04-30', 'Industrials')
generate_report('./', 'Consumer Staples_TransactionHistory2020-2022_withCash', '2020-04-30', 'Consumer Staples')
generate_report('./', 'Utilities_TransactionHistory2020-2022_withCash', '2020-04-30', 'Utilities')
generate_report('./', 'Materials_TransactionHistory2020-2022_withCash', '2020-04-30', 'Materials')
generate_report('./', 'Energy_TransactionHistory2020-2022_withCash', '2020-04-30', 'Energy')
'''

