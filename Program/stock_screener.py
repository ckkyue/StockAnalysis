# Imports
import ast
import concurrent.futures
import datetime as dt
from fundamentals import *
from helper_functions import get_current_date, generate_end_dates, get_currency, get_df, get_earning_dates, get_excel_filename, get_infix, get_rs_volume, slope_reg, stock_market
import numpy as np
import pandas as pd
from pandas import ExcelWriter as EW
import os
from scipy.stats import linregress
from sklearn.preprocessing import MinMaxScaler
from technicals import *
from tqdm import tqdm
import yfinance as yf

# Calculate the stop loss and target price of a stock
def stoploss_target(stock, entry, end_date, period=5, max_stoploss=0.08, atr_buffer=0.25, rr=2):
    # Get the price data of the stock
    df = get_df(stock, end_date)

    # Filter the data
    df = df[df.index <= end_date]

    # Calculate the minimum lowest price over the past period
    low_min = df["Low"].rolling(window=period).min().iloc[-1]
    
    # Calculate the average true range (ATR)
    atr = ATR(df)["ATR"].iloc[-1]

    # Calculate the stop loss
    stoploss = max((1 - max_stoploss) * entry, low_min - atr_buffer * atr)

    # Calculate the stop loss percentage
    stoploss_pct = (1 - stoploss / entry) * 100

    # Calculate the target price
    target = entry + (entry - stoploss) * rr

    # Calculate the target price percentage
    target_pct = (target / entry - 1) * 100

    # Round the values
    stoploss = round(stoploss, 2)
    stoploss_pct = round(stoploss_pct, 1)
    target = round(target, 2)
    target_pct = round(target_pct, 1)
    
    return stoploss, stoploss_pct, target, target_pct

# Create dataframes to store RS ratings and volume ranks
def create_rs_volume_df(tickers, current_date, end_dates, periods, index_returns, index_shortName, result_folder, infix, backtest, print_multiple=True):
    # Convert inputs to lists
    if not isinstance(end_dates, list):
        end_dates = [end_dates]
    if not isinstance(periods, list):
        periods = [periods]
    if not isinstance(index_returns, list):
        index_returns = [index_returns]

    # Initialize three empty lists to store rs_df, volume_df, and rs_volume_df
    rs_dfs = []
    volume_dfs = []
    rs_volume_dfs = []

    # Fetch data once per ticker
    dfs = {}
    for ticker in tqdm(tickers, desc="Fetching ticker data"):
        dfs[ticker] = get_df(ticker, current_date)
    
    # Iterate over all combinations of end dates, periods, and index return
    for end_date, period, index_return in zip(end_dates, periods, index_returns):
        return_muls = {}
        volume_smas = {}

        # Iterate over all tickers
        for ticker in tqdm(tickers, desc=f"Processing data for {end_date}"):
            try:
                df = dfs.get(ticker)
                if df is None:
                    continue
                
                # Filter the data
                df = df[df.index <= end_date]

                # Calculate the percent change of the stock
                df["Percent Change"] = df["Close"].pct_change()

                # Calculate the stock return
                stock_return = (df["Percent Change"] + 1).tail(period).cumprod().iloc[-1]

                # Calculate the stock return relative to the market
                return_mul = stock_return / index_return
                return_muls[ticker] = return_mul
                if print_multiple:
                    print(f"Ticker: {ticker} ; Return multiple against {index_shortName}: {round(return_mul, 2)}\n")
                
                # Calculate the moving averages of volume
                df["Volume SMA 5"] = SMA(df, 5, column="Volume")
                df["Volume SMA 20"] = SMA(df, 20, column="Volume")
                volume_smas[ticker] = {"Volume SMA 5": df["Volume SMA 5"].iloc[-1], "Volume SMA 20": df["Volume SMA 20"].iloc[-1]}

            except Exception as e:
                print(f"Error processing data for {ticker}: {e}\n")
                continue

            # time.sleep(0.05)
            
        # Create a dataframe to store the RS ratings of tickers
        return_muls = dict(sorted(return_muls.items(), key=lambda x: x[1], reverse=True))
        rs_df = pd.DataFrame(return_muls.items(), columns=["Ticker", "Value"])
        rs_df["RS"] = rs_df["Value"].rank(pct=True) * 100
        rs_df = rs_df[["Ticker", "RS"]]

        # Create a dataframe to store the volume ranks of tickers
        volume_df = pd.DataFrame.from_dict(volume_smas, orient="index", columns=["Volume SMA 5", "Volume SMA 20"])
        volume_df["Ticker"] = volume_df.index
        volume_df.reset_index(drop=True, inplace=True)
        volume_df["Volume SMA 5 Rank"] = volume_df["Volume SMA 5"].rank(ascending=False)
        volume_df["Volume SMA 20 Rank"] = volume_df["Volume SMA 20"].rank(ascending=False)

        # Merge the dataframes
        rs_volume_df = pd.merge(rs_df, volume_df, on="Ticker")
        rs_volume_df = rs_volume_df.sort_values(by="RS", ascending=False)

        # Check if there are pre-existing data
        current_files = [file for file in os.listdir(result_folder) if file.startswith(f"{infix}rsvolume_")]

        # Get the list of dates
        dates = [file.split("_")[-1].replace(".csv", "") for file in current_files]

        # Remove the old files for dates prior to the end date
        for date in dates:
            if date < end_date:
                os.remove(os.path.join(result_folder, f"{infix}rsvolume_{date}.csv"))
                
        # Define the filename
        filename = os.path.join(result_folder, f"{infix}rsvolume_{end_date}.csv")

        # Save the merged dataframe to a .csv file
        if not backtest:
            rs_volume_df.to_csv(filename, index=False)

        rs_dfs.append(rs_df)
        volume_dfs.append(volume_df)
        rs_volume_dfs.append(rs_volume_df)

    if len(rs_dfs) == 1:
        return rs_dfs[0], volume_dfs[0], rs_volume_dfs[0]
    else:
        return rs_dfs, volume_dfs, rs_volume_dfs

# Combine the long term and short term RS dataframes
def longshortRS(stocks, index_df, index_name, index_dict, NASDAQ_all, current_date, end_dates1, end_dates2, periods1, periods2, result_folder, volume_filter=None):
    # Get the infix
    infix = get_infix(index_name, index_dict, NASDAQ_all)

    # Convert inputs to lists
    if not isinstance(end_dates1, list):
        end_dates1 = [end_dates1]
    if not isinstance(end_dates2, list):
        end_dates2 = [end_dates2]
    if not isinstance(periods1, list):
        periods1 = [periods1]
    if not isinstance(periods2, list):
        periods2 = [periods2]
        
    # Initialize an empty list to store the merged dataframes
    merged_dfs = []

    # Initialize an empty list to store the index returns
    index_returns = []

    # Iterate over all combinations
    for end_date1, end_date2, period1, period2 in zip(end_dates1, end_dates2, periods1, periods2):
        # Filter the data
        index_df1 = index_df[index_df.index <= end_date1]
        index_df2 = index_df[index_df.index <= end_date2]

        # Calculate the percent change of the index
        index_df1.loc[:, "Percent Change"] = index_df1["Close"].pct_change()
        index_df2.loc[:, "Percent Change"] = index_df2["Close"].pct_change()
        
        # Calculate the total return of the index
        index_return1 = (index_df1["Percent Change"] + 1).tail(period1).cumprod().iloc[-1]
        index_return2 = (index_df2["Percent Change"] + 1).tail(period2).cumprod().iloc[-1]
        index_shortName = index_dict[f"{index_name}"]
        print(f"Return for {index_shortName} between {index_df1.index[-period1].strftime('%Y-%m-%d')} and {end_date1}: {index_return1:.2f}")
        print(f"Return for {index_shortName} between {index_df2.index[-period2].strftime('%Y-%m-%d')} and {end_date2}: {index_return2:.2f}")

        index_returns.extend([index_return1, index_return2])
        
    rs_dfs, volume_dfs, _ = create_rs_volume_df(stocks, current_date, end_dates1 + end_dates2, periods1 + periods2, index_returns, index_shortName, result_folder, infix, True, print_multiple=False)

    # Separate the dataframes into two halves
    length_df = len(rs_dfs) // 2
    rs_dfs1, rs_dfs2 = rs_dfs[:length_df], rs_dfs[length_df:]
    volume_dfs1, volume_dfs2 = volume_dfs[:length_df], volume_dfs[length_df:]

    for rs_df1, rs_df2, volume_df1, volume_df2 in zip(rs_dfs1, rs_dfs2, volume_dfs1, volume_dfs2):
        if volume_filter is not None:
            volume_df1 = volume_df1[(volume_df1["Volume SMA 5 Rank"] <= volume_filter) | (volume_df1["Volume SMA 20 Rank"] <= volume_filter)]
            volume_df2 = volume_df2[(volume_df2["Volume SMA 5 Rank"] <= volume_filter) | (volume_df2["Volume SMA 20 Rank"] <= volume_filter)]

            # Filter rs_df1 and rs_df2 based on the tickers present in volume dataframes
            rs_df1 = rs_df1[rs_df1["Ticker"].isin(set(volume_df1["Ticker"]))]
            rs_df2 = rs_df2[rs_df2["Ticker"].isin(set(volume_df2["Ticker"]))]

        # Merge and clean data
        merged_df = pd.merge(rs_df1, rs_df2, on="Ticker", suffixes=(" 1", " 2"))
        merged_df = merged_df.rename(columns={"RS 1": "Long-term RS", "RS 2": "Short-term RS"}).dropna()
        merged_dfs.append(merged_df)

    return merged_dfs[0] if len(merged_dfs) == 1 else merged_dfs

# Compare the long and short term RS
def compare_longshortRS(stocks, index_df, index_name, index_dict, NASDAQ_all, current_date, end_dates, period1, period2, result_folder):
    # Initialize two empty lists to store the RS slopes and R^2 values
    rs_slopes = []
    r_squareds = []

    # Define the end dates and periods
    end_dates1 = []
    end_dates2 = []
    for i in range(len(end_dates) - 1):
        end_date = end_dates[i]
        end_dates1.append(end_date)
        end_dates2.append((dt.datetime.strptime(end_date, "%Y-%m-%d") + relativedelta(days=20)).strftime("%Y-%m-%d"))
    periods1 = [period1] * len(end_dates1)
    periods2 = [period2] * len(end_dates2)

    # Get the merged dataframe
    merged_dfs = longshortRS(stocks, index_df, index_name, index_dict, NASDAQ_all, current_date, end_dates1, end_dates2, periods1, periods2, result_folder)
    
    # Iterate over merged dataframe
    for merged_df in merged_dfs:
        # Calculate the slope and R^2
        rs_slope, _, r_value, _, _ = linregress(merged_df["Long-term RS"], merged_df["Short-term RS"])
        r_squared = r_value**2
        rs_slopes.append(rs_slope)
        r_squareds.append(r_squared)
        
    return rs_slopes, r_squareds, end_dates2

# Get the information of a stock from yfinance
def get_stock_info(stock):
    try:
        return yf.Ticker(stock).info
    
    except Exception as e:
        print((f"Error for get_stock_info {stock}: {e}\n"))

        return None
    
# Define a function to choose between SMA and EMA
def EMA_replace(SMA_value, EMA_value):
    return EMA_value if np.isnan(SMA_value) else SMA_value

# Check if the price data fulfills the technical conditions
def check_conds_tech(index_name, current_close, SMA_20, SMA_20_slope, SMA_50, SMA_50_slope, SMA_200, SMA_200_slope, EMA_20, EMA_20_slope, EMA_50, EMA_50_slope, EMA_200, EMA_200_slope, Low, High):
    # Technicals
    if index_name == "^HSI":
        cond_t1 = current_close > EMA_replace(SMA_20, EMA_20) > EMA_replace(SMA_50, EMA_50)
        cond_t2 = current_close > EMA_replace(SMA_200, EMA_200)
        cond_t3 = EMA_replace(SMA_20_slope, EMA_20_slope) > 0
        conds_tech = cond_t1 and cond_t2 and cond_t3
        
        return conds_tech, cond_t1, cond_t2, cond_t3

    else:
        cond_t1 = current_close > EMA_replace(SMA_50, EMA_50) > EMA_replace(SMA_200, EMA_200)
        cond_t2 = EMA_replace(SMA_50_slope, EMA_50_slope) > 0
        cond_t3 = EMA_replace(SMA_200_slope, EMA_200_slope) > 0
        cond_t4 = current_close >= (1.25 * Low)
        cond_t5 = current_close >= (0.75 * High)
        conds_tech = cond_t1 and cond_t2 and cond_t3 and cond_t4 and cond_t5

        return conds_tech, cond_t1, cond_t2, cond_t3, cond_t4, cond_t5
    
# Check if the stock fulfills the fundamental conditions
def check_conds_fund(Y_growth, Q_growth, ROE):
    try:
        cond_f2 = Y_growth >= 0
    except Exception:
        cond_f2 = False
    try:
        cond_f3 = Q_growth >= 0
    except Exception:
        cond_f3 = False
    try:
        cond_f4 = ROE >= 0
    except Exception:
        cond_f4 = False
    
    conds_fund = cond_f2 and cond_f3 and cond_f4

    return conds_fund, cond_f2, cond_f3, cond_f4

# Check the Minervini conditions for the top performing stocks
def process_stock(stock, index_name, end_date, current_date, stock_data, stock_info_data, rs_volume_df, backtest=False):
    # Get the currency
    currency = get_currency(index_name)

    try:
        # Get the data and information of the stock
        df = stock_data[stock]
        stock_info = stock_info_data[stock]

        # Preprocess stock data
        # Filter the data
        df = df[df.index <= end_date]

        # Current closing price
        current_close = df["Close"].iloc[-1]

        # Calculate the moving averages
        periods = [5, 20, 50, 200]

        for i in periods:
            df[f"SMA {str(i)}"] = SMA(df, i)
            df[f"EMA {str(i)}"] = EMA(df, i)

        # Calculate the moving averages
        SMA_5 = df["SMA 5"].iloc[-1]
        SMA_20 = df["SMA 20"].iloc[-1]
        SMA_50 = df["SMA 50"].iloc[-1]
        SMA_200 = df["SMA 200"].iloc[-1]
        SMA_20_slope = df["SMA 20"].diff().iloc[-1]
        SMA_50_slope = df["SMA 50"].diff().iloc[-1]
        SMA_200_slope = df["SMA 200"].diff().iloc[-1]
        EMA_5 = df["EMA 5"].iloc[-1]
        EMA_20 = df["EMA 20"].iloc[-1]
        EMA_50 = df["EMA 50"].iloc[-1]
        EMA_200 = df["EMA 200"].iloc[-1]
        EMA_20_slope = df["EMA 20"].diff().iloc[-1]
        EMA_50_slope = df["EMA 50"].diff().iloc[-1]
        EMA_200_slope = df["EMA 200"].diff().iloc[-1]

        # 52 week Low
        Low = round(min(df["Low"][-252:]), 2)

        # 52 week High
        High = round(max(df["High"][-252:]), 2)

        # RS rating and volume SMA 5 rank
        RS_rating, volume_sma5_rank = get_rs_volume(stock, rs_volume_df)

        # Check the Minervini conditions
        # Technicals
        if index_name == "^HSI":
            conds_tech, cond_t1, cond_t2, cond_t3 = check_conds_tech(index_name, current_close, SMA_20, SMA_20_slope, SMA_50, SMA_50_slope, SMA_200, SMA_200_slope, EMA_20, EMA_20_slope, EMA_50, EMA_50_slope, EMA_200, EMA_200_slope, Low, High)
        else:
            conds_tech, cond_t1, cond_t2, cond_t3, cond_t4, cond_t5 = check_conds_tech(index_name, current_close, SMA_20, SMA_20_slope, SMA_50, SMA_50_slope, SMA_200, SMA_200_slope, EMA_20, EMA_20_slope, EMA_50, EMA_50_slope, EMA_200, EMA_200_slope, Low, High)

        # Preprocess stock information
        if conds_tech:
            market_cap = get_market_cap(stock, stock_info, end_date, current_date)

            # Fundamentals
            cond_f1 = market_cap != "N/A" and market_cap > 1

            # Check if the conditions are met
            if conds_tech and cond_f1:
                # Get the trailing and forward EPS
                tEPS = stock_info.get("trailingEps", "N/A")
                fEPS = stock_info.get("forwardEps", "N/A")

                # Estimate the EPS growth of next year
                EPS_nextY_growth = round((fEPS - tEPS) / np.abs(tEPS) * 100, 2) if tEPS != "N/A" else "N/A"
                
                if index_name == "^HSI":
                    # Get the earnings growth of the most recent quarters
                    earnings_thisQ_growth = stock_info["earningsQuarterlyGrowth"] * 100

                    # Get the ROE
                    ROE = stock_info["returnOnEquity"] * 100

                else:
                    EPS_past5Y_growth, EPS_thisY_growth, EPS_QoQ_growth, ROE = get_fundamentals(stock, end_date, current_date)

                if index_name == "^HSI":
                    conds_fund, cond_f2, cond_f3, cond_f4 = check_conds_fund(EPS_nextY_growth, earnings_thisQ_growth, ROE)
                else:
                    conds_fund, cond_f2, cond_f3, cond_f4 = check_conds_fund(EPS_thisY_growth, EPS_QoQ_growth, ROE)
                
                if conds_fund:
                    # Get the sector and industry of the stock
                    sector = stock_info.get("sector", "N/A")
                    industry = stock_info.get("industry", "N/A")

                    # Get the quarterly growths of the stock
                    EPS_thisQ_growth, EPS_last1Q_growth, EPS_last2Q_growth = get_lastQ_growths(stock, index_name, end_date, current_date)

                    # Calculate the volatility of the stock over past 1 month
                    data = get_volatility(df)
                    volatility_20 = data["Volatility 20"].iloc[-1]
                    volatility_60 = data["Volatility 60"].iloc[-1]
                    
                    # MVP/VCP condition
                    data = MVP_VCP(df)
                    MVP = data["MVP"].iloc[-1]
                    M_past60 = data["M past 60"].iloc[-1]
                    MV_past60 = data["MV past 60"].iloc[-1]
                    MP_past60 = data["MP past 60"].iloc[-1]
                    MVP_past60 = data["MVP past 60"].iloc[-1]
                    MVP_rating = data["MVP Rating"].iloc[-1]
                    VCP = data["VCP"].iloc[-1]
                    pivot_breakout = data["Pivot breakout"].iloc[-1]
                    volume_shrink = data["Volume shrinking"].iloc[-1]

                    # Get the next earning date
                    try:
                        earning_dates = get_earning_dates(stock)
                        next_earning_date = str(earning_dates[earning_dates > end_date].min())

                    except Exception as e:
                        print(f"Error getting next earning date {stock}: {e}\n")
                        next_earning_date = "N/A"

                    # Relevant information of the stock
                    result = {
                        "Stock": stock,
                        "RS Rating": RS_rating,
                        "Volume SMA 5 Rank": volume_sma5_rank,
                        "Close": round(current_close, 2),
                        "Volatility 20 (%)": round(volatility_20 * 100, 2),
                        "Volatility 60 (%)": round(volatility_60 * 100, 2),
                        "SMA 5": EMA_replace(SMA_5, EMA_5),
                        "SMA 20": EMA_replace(SMA_20, EMA_20),
                        "SMA 50": EMA_replace(SMA_50, EMA_50),
                        "SMA 200": EMA_replace(SMA_200, EMA_200),
                        "SMA 5/20 Ratio": round(SMA_5 / SMA_20, 2),
                        "SMA 5/50 Ratio": round(SMA_5 / SMA_50, 2),
                        "MVP": MVP,
                        "M past 60": M_past60,
                        "MV past 60": MV_past60,
                        "MP past 60": MP_past60,
                        "MVP past 60": MVP_past60,
                        "MVP Rating": MVP_rating,
                        "VCP": VCP,
                        "Pivot Preakout": pivot_breakout,
                        "Volume Shrinking": volume_shrink,
                        "52 Week Low": Low,
                        "52 Week High": High,
                        f"Market Cap (B, {currency})": market_cap,
                        "EPS past 5Y (%)": EPS_past5Y_growth if index_name != "^HSI" else "N/A",
                        "EPS this Y (%)": EPS_thisY_growth if index_name != "^HSI" else "N/A",
                        "EPS Q/Q (%)": EPS_QoQ_growth if index_name != "^HSI" else "N/A",
                        "ROE (%)": ROE,
                        "EPS this Q (%)": EPS_thisQ_growth if index_name != "^HSI" else "N/A",
                        "EPS last 1Q (%)": EPS_last1Q_growth if index_name != "^HSI" else "N/A",
                        "EPS last 2Q (%)": EPS_last2Q_growth if index_name != "^HSI" else "N/A",
                        "Next Earning Date": next_earning_date,
                        "Sector": sector,
                        "Industry": industry,
                    }
                    if not backtest:
                        result.update({
                            "Trailing EPS": tEPS,
                            "Forward EPS": fEPS,
                            "Estimated EPS growth (%)": EPS_nextY_growth,
                        })

                    if index_name == "^HSI":
                        result.update({
                            "Earnings this Q (%)": earnings_thisQ_growth,
                        })

                    return result
                
    except Exception as e:
        print(f"Error for {stock}: {e}\n")

        return None
    
# Calculate the EM rating
def EM_rating(index_name, data, factors):
    # Define the target columns based on index name
    if index_name == "^HSI":
        target_columns = ["MVP Rating", "Estimated EPS growth (%)", "Earnings this Q (%)"]
    else:
        target_columns = ["MVP Rating", "EPS this Y (%)", "EPS Q/Q (%)"]

    data_copy = data.copy()

    # Extract the number of stocks
    stocks_num = data_copy.shape[0]

    # Skip if the number of stocks is less than or equal to 1
    if stocks_num <= 1:
        return data

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # Normalize the first column
    data_copy[target_columns[0]] = scaler.fit_transform(data_copy[target_columns[0]].values.reshape(-1, 1))

    # Apply log1p and MinMaxScaler to the last two columns
    for column in target_columns[1:]:
        min_value = data_copy[column].min()
        if min_value < 0:
            # Minus the minimum value before applying log1p
            data_copy[column] = np.log1p(data_copy[column] - min_value)
        else:
            data_copy[column] = np.log1p(data_copy[column])
            
        # Normalize the last two columns
        data_copy[column] = scaler.fit_transform(data_copy[column].values.reshape(-1, 1))

    # Calculate the weighted average for each row and multiply by 100
    data["EM Rating"] = (data_copy[target_columns] * factors / np.sum(factors)).sum(axis=1) * 100

    # Sort the EM ratings in descending order
    data = data.sort_values("EM Rating", ascending=False)
    
    return data

# Select the stocks
def select_stocks(end_dates, current_date, index_name, index_dict, 
                  period_hk, period_us, RS, NASDAQ_all, factors, backtest):
    # Select period based on HK/US
    if index_name == "^HSI":
        period = period_hk
    else:
        period = period_us

    # Get the infix
    infix = get_infix(index_name, index_dict, NASDAQ_all)

    # Define the result folder
    result_folder = "Result"

    # Iterate over all end dates
    for end_date in end_dates.copy():
        # Define the filename
        filename = get_excel_filename(end_date, index_name, index_dict, period_hk, period_us, RS, NASDAQ_all, result_folder)

        # Remove the end date if the file exists
        if os.path.isfile(filename):
            end_dates.remove(end_date)

    # Iterate over all end dates
    for end_date in tqdm(end_dates):
        # Get the tickers of the stock market
        tickers = stock_market(end_date, current_date, index_name, NASDAQ_all)
        
        # Get the price data of the index
        index_df = get_df(index_name, current_date)

        # Filter the data
        index_df = index_df[index_df.index <= end_date]

        # Calculate the percent change of the index
        index_df["Percent Change"] = index_df["Close"].pct_change()

        # Calculate the total return of the index
        index_return = (index_df["Percent Change"] + 1).tail(period).cumprod().iloc[-1]
        index_shortName = index_dict[f"{index_name}"]
        print(f"Return for {index_shortName} between {index_df.index[-period].strftime('%Y-%m-%d')} and {end_date}: {index_return:.2f}")

        # Find the return multiples and volumes
        rs_df, volume_df, rs_volume_df = create_rs_volume_df(tickers, current_date, end_date, period, index_return, index_shortName, result_folder, infix, backtest)
        
        # Filter the stocks
        if index_name == "^HSI":
            volume_df = volume_df[(volume_df["Volume SMA 5 Rank"] <= 200) | (volume_df["Volume SMA 20 Rank"] <= 200)]
            stocks = volume_df["Ticker"]
        else:
            rs_df = rs_df[rs_df["RS"] >= RS]
            stocks = rs_df["Ticker"]

        # Fetch the stock data and stock information in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            stock_data = {stock: data for stock, data in zip(stocks, executor.map(lambda stock: get_df(stock, end_date), stocks))}
            stock_info_data = {stock: info for stock, info in zip(stocks, executor.map(get_stock_info, stocks))}

        # Process each stock and create an export list
        export_data = [process_stock(stock, index_name, end_date, current_date, stock_data, stock_info_data, rs_volume_df, backtest=backtest) for stock in tqdm(stocks)]
        export_data = [row for row in export_data if row is not None]
        df = pd.DataFrame(export_data)
        df = EM_rating(index_name, df, factors)

        # Calculate the means and standard deviations
        volatility_20_mean = df["Volatility 20 (%)"].mean()
        volatility_60_mean = df["Volatility 60 (%)"].mean()
        volatility_20_sd = df['Volatility 20 (%)'].std()
        volatility_60_sd = df['Volatility 60 (%)'].std()

        # Calculate the z-scores
        volatility_20_zscore = (df["Volatility 20 (%)"] - volatility_20_mean) / volatility_20_sd
        volatility_60_zscore = (df["Volatility 60 (%)"] - volatility_60_mean) / volatility_60_sd

        # Insert the z-scores
        df.insert(df.columns.get_loc("Volatility 20 (%)") + 1, "Volatility 20 Z-Score", volatility_20_zscore)
        df.insert(df.columns.get_loc("Volatility 60 (%)") + 1, "Volatility 60 Z-Score", volatility_60_zscore)

        # Format the end date
        end_date_fmt = dt.datetime.strptime(end_date, "%Y-%m-%d").strftime("%d-%m-%y")

        # Check if the "end_date_fmt" folder exists inside the "Result" folder, create it if it does not
        folder_path = os.path.join(result_folder, f"{end_date_fmt}")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Export the results to an Excel file inside the "end_date_fmt" folder
        filename = get_excel_filename(end_date, index_name, index_dict, period_hk, period_us, RS, NASDAQ_all, result_folder)
        writer = EW(filename)
        df.to_excel(writer, sheet_name="Sheet1", index=False)
        writer._save()

# Create the stock dictionary
def create_stock_dict(end_dates, index_name, index_dict, NASDAQ_all, factors, top=10, RS=90, period=252):
    # Get the infix
    infix = get_infix(index_name, index_dict, NASDAQ_all)

    # Initialize stock_dict
    stock_dict = {}

    # Check if stock_dict exists
    stock_dict_filename = f"Result/Stock dict/{infix}stock_dict{factors}.txt"
    if os.path.isfile(stock_dict_filename):
        with open(stock_dict_filename, "r") as file:
            # Retrieve the content of the stock_dict as a dictionary
            stock_dict = ast.literal_eval(file.read())

    # Iterate over all end dates
    for end_date in end_dates[:-1]:
        # Format the end date
        end_date_fmt = dt.datetime.strptime(end_date, "%Y-%m-%d").strftime("%d-%m-%y")
        filename = f"Result/{end_date_fmt}/{infix}stock_{end_date_fmt}period{period}RS{RS}.xlsx"

        # Read the data of the screened stocks
        df = pd.read_excel(filename)

        # Calculate the EM rating
        df = EM_rating(df, factors)

        # Extract the number of stocks
        stocks_num = df.shape[0]

        # Return None if the number of stocks is 0
        if stocks_num == 0:
            stock_dict[end_date] = None
        else:
            # Extract the stocks with top EM ratings
            top_stocks = df.head(top)["Stock"].tolist()
            stock_dict[end_date] = top_stocks

        # Sort tock_dict based on the ascending order of dates
        stock_dict = dict(sorted(stock_dict.items(), key=lambda x: dt.datetime.strptime(x[0], "%Y-%m-%d")))

    # Open the file in write mode
    with open(stock_dict_filename, "w") as file:
        # Write the dictionary string to the file
        file.write(str(stock_dict))

# Main function
def main():
    # Start of the program
    start = dt.datetime.now()
    print(start, "\n")

    # Define the paths for the folders
    folders = ["Price data"]
    
    # Check if the folders exist, create them if they do not
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    # Variables
    HKEX_all = False
    NASDAQ_all = True
    period_hk = 60 # Period for HK stocks
    period_us = 252 # Period for US stocks
    RS = 90
    factors = [1, 1, 1]
    backtest = False

    # Index
    index_name = "^GSPC"
    index_dict = {"^HSI": "HKEX", "^GSPC": "S&P 500", "^IXIC": "NASDAQ Composite"}

    # Get the current date
    current_date = get_current_date(start, index_name)

    # Create the end dates
    end_dates = generate_end_dates(5, current_date)
    end_dates.append(current_date)
    end_dates = [current_date]

    # Stock selection
    select_stocks(end_dates, current_date, index_name, index_dict, 
                  period_hk, period_us, RS, NASDAQ_all, factors, backtest)

    # Print the end time and total runtime
    end = dt.datetime.now()
    print(end, "\n")
    print("The program used", end - start)

# Run the main function
if __name__ == "__main__":
    main()