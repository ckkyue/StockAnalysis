# Imports
import datetime as dt
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import os
from scipy.stats import linregress
from yahoo_fin import stock_info as si
import yfinance as yf

# Get the current date
def get_current_date(start):
    # Check if today is Sunday
    if start.weekday() == 6:
        # Revert back one day
        current_date = (start - dt.timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        current_date = start.strftime("%Y-%m-%d")
    
    return current_date

# Get the price data of a stock
def get_df(stock, end_date, redownload=False):
    # Initial setup
    csv_date = (dt.datetime.strptime(end_date, "%Y-%m-%d") - relativedelta(years=40)).strftime("%Y-%m-%d")

    # Define the folder path
    folder_path = "Price data"

    # Check if there are pre-existing data
    current_files = [file for file in os.listdir(folder_path) if file.startswith(f"{stock}_")]

    # Get the list of dates
    dates = [file.split("_")[-1].replace(".csv", "") for file in current_files]

    # Get the maximum date from the list of dates
    max_date = max(dates) if dates else "N/A"

    # Remove the old files for dates prior to the maximum date
    if max_date != "N/A":
        for date in dates:
            if date < max_date:
                os.remove(os.path.join(folder_path, f"{stock}_{date}.csv"))
        # Define the filename
        if end_date >= max_date:
            filename = os.path.join(folder_path, f"{stock}_{end_date}.csv")
        else:
            filename = os.path.join(folder_path, f"{stock}_{max_date}.csv")
    else:
        filename = os.path.join(folder_path, f"{stock}_{end_date}.csv")
    
    # Save the price data to a .csv file if the most updated data do not exist
    if not os.path.isfile(filename) or redownload:
        df = yf.download(stock, start=csv_date, end=end_date)
        if not df.empty:
            df.to_csv(filename)
            df = pd.read_csv(filename)

            # Remove the old file for the maximum date
            if max_date != "N/A":
                if max_date < end_date:
                    os.remove(os.path.join(folder_path, f"{stock}_{max_date}.csv"))
                    
        else:
            try:
                print(f"The price data of {stock} cannot be updated.")
            except Exception as e:
                print(f"Error for {stock}: {e}.")
                
            return None

    # Read the most updated data
    else:
        df = pd.read_csv(filename)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    return df

# Generate a list of end dates
def generate_end_dates(years, current_date):
    # Calculate the last end date
    current = dt.datetime.strptime(current_date, "%Y-%m-%d")
    this_month = current.month
    last_end_date = current.replace(year=current.year, month=this_month, day=1).strftime("%Y-%m-%d")

    # Get the price data of the index
    df = get_df("^GSPC", current_date)

    # Retrieve the trading dates at 1-month intervals
    last_end_date = dt.datetime.strptime(last_end_date, "%Y-%m-%d")
    end_dates = [df.index[df.index <= last_end_date - relativedelta(months=i)].max().strftime("%Y-%m-%d") for i in range(years * 12 + this_month - 1, -1, -1)]

    return end_dates

# Get the earning dates of using yfinance
def get_earning_dates(stock):
    years = 30
    earning_dates = yf.Ticker(stock).earnings_dates
    df = pd.DataFrame(earning_dates)
    earning_dates = df.index
    last_earning_date = earning_dates[-1]

    # Iterate over all quarters of past years
    for i in range((years * 4)):
        earning_dates = earning_dates.append(pd.Index([last_earning_date - relativedelta(months=i*3)]))

    return earning_dates

# Get the Excel filename
def get_excel_filename(end_date, index_name, index_dict, period_hk, period_us, RS, NASDAQ_all, result_folder):
        # Select period based on HK/US
        if index_name == "^HSI":
            period = period_hk
        else:
            period = period_us

        # Get the infix
        infix = get_infix(index_name, index_dict, NASDAQ_all)

        # Format the end date
        end_date_fmt = dt.datetime.strptime(end_date, "%Y-%m-%d").strftime("%d-%m-%y")

        # Define the folder path
        folder_path = os.path.join(result_folder, end_date_fmt)

        # Define the Excel filename
        excel_filename = os.path.join(folder_path, f"{infix}stock_{end_date_fmt}period{period}RS{RS}.xlsx")

        return excel_filename

# Merge dataframes of stocks
def merge_stocks(stocks, end_date):
    # Get the price data of the stocks
    dfs = [get_df(stock, end_date) for stock in stocks]

    # Rename the columns of the first DataFrame to include the stock name
    df_merged = dfs[0].rename(columns=lambda col: f"{col} ({stocks[0]})")

    # Join the remaining DataFrames with the appropriate suffix
    for i in range(1, len(dfs)):
        df_merged = df_merged.join(dfs[i].rename(columns=lambda col: f"{col} ({stocks[i]})"), how="inner")

    return df_merged

# Get the list of tickers of stock market
def stock_market(end_date, current_date, index_name, NASDAQ_all):
    # HKEX
    if index_name == "^HSI":
        hkex_df = pd.read_excel("Program/ListOfSecurities.xlsx", skiprows=2)
        hkex_df = hkex_df[hkex_df["Category"] == "Equity"]
        stocks = hkex_df["Stock Code"].tolist()
        tickers = [str(int(stock)).zfill(4) + '.HK' for stock in stocks]

    # S&P 500
    elif not NASDAQ_all and index_name == "^GSPC":
        # Read the .csv file containing historical components of S&P 500
        sp500_df = pd.read_csv("Program/sp_500_historical_components.csv")
        sp500_df["date"] = pd.to_datetime(sp500_df["date"])
        sp500_df.set_index("date", inplace=True)

        # Read Wikipedia to get the current components of S&P 500
        tickers_table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        tickers = tickers_table["Symbol"].tolist()
        tickers = [str(ticker).replace(".", "-").replace("^", "-P").replace("/", "-") for ticker in tickers]
        tickers.sort()
        
        # Save the tickers to the .csv file
        sp500_df.loc[pd.to_datetime(current_date), "tickers"] = ",".join(tickers)
        sp500_df.to_csv("Program/sp_500_historical_components.csv")

        # Get the list of tickers
        tickers = sp500_df[sp500_df.index <= end_date]["tickers"].iloc[-1].split(",")
        tickers = [str(ticker).replace(".", "-").replace("^", "-P").replace("/", "-") for ticker in tickers]
        tickers.sort()

    # NASDAQ Composite
    elif index_name == "^IXIC":
        tickers = si.tickers_nasdaq()
        tickers = [str(ticker).replace(".", "-").replace("^", "-P").replace("/", "-") for ticker in tickers]
        tickers.sort()

    # NASDAQ and NYSE (NASDAQ_all)
    elif NASDAQ_all and index_name == "^GSPC":
        tickers_nasdaq = pd.read_csv("Program/nasdaq.csv")
        tickers_nyse = pd.read_csv("Program/nyse.csv")
        tickers_table = pd.concat([tickers_nasdaq, tickers_nyse]).drop_duplicates(subset="Symbol", keep="first")
        tickers = tickers_table["Symbol"].tolist()
        tickers = [str(ticker).replace(".", "-").replace("^", "-P").replace("/", "-") for ticker in tickers]
        tickers.sort()
        
    return tickers

# Get the infix
def get_infix(index_name, index_dict, NASDAQ_all):
    if NASDAQ_all and index_name == "^GSPC":
        infix = "NASDAQ"
    else:
        infix = index_dict[index_name].replace(" ", "")
        
    return infix

# Get the currency
def get_currency(index_name):
    if index_name == "^HSI":
        currency = "HKD"
    elif index_name == "^GSPC" or "^IXIC":
        currency = "USD"

    return currency

# Find the RS rating and volume SMA 5 rank of a ticker
def get_rs_volume(ticker, rs_volume_df):
    if ticker in rs_volume_df["Ticker"].values:
        row = rs_volume_df.loc[rs_volume_df["Ticker"] == ticker]
        rs = row["RS"].iloc[0]
        volume_sma5_rank = row["Volume SMA 5 Rank"].iloc[0]

        return rs, volume_sma5_rank
    else:
        return None

# Slope function
def slope_reg(arr):
    y = np.array(arr)
    x = np.arange(len(y))
    slope = linregress(x, y)[0]
    
    return slope

# Randomize an array
def randomize_array(arr):
    # Get the length of the array
    length = len(arr)
    
    # Randomize the array by swapping two random elements 10 times
    for i in range(10):
        index = np.random.randint(0, length - 1)
        arr[index], arr[index + 1] = arr[index + 1], arr[index]

    return arr * np.random.uniform(low=0.8, high=1.2, size=arr.shape)