# Imports
import ast
import concurrent.futures
import datetime as dt
from fundamentals import *
from helper_functions import generate_end_dates, get_currency, get_df, get_earning_dates, get_infix, get_rs_volume, slope_reg, stock_market
import numpy as np
import pandas as pd
from pandas import ExcelWriter as EW
import os
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

# Get the data of a stock
def get_stock_data(stock, end_date, current_date):
    try:
        # Read the price data of the stock
        df = pd.read_csv(f"Price data/{stock}_{current_date}.csv", index_col=0)

        # Filter the data
        df = df[df.index <= end_date]

        # Calculate the moving averages
        periods = [5, 20, 50, 200]

        try:
            for i in periods:
                df[f"SMA {str(i)}"] = SMA(df, i)
        except Exception:
            for i in periods:
                df[f"EMA {str(i)}"] = EMA(df, i)

        return df
    except Exception as e:
        print((f"Error for get_stock_data {stock}: {e}\n"))

        return None

# Get the information of a stock from yfinance
def get_stock_info(stock):
    try:
        return yf.Ticker(stock).info
    except Exception as e:
        print((f"Error for get_stock_info {stock}: {e}\n"))

        return None

# Check the Minervini conditions for the top performing stocks
def process_stock(stock, index_name, end_date, current_date, stock_data, stock_info_data, rs_volume_df, backtest=False):
    # Get the currency
    currency = get_currency(index_name)

    try:
        # Get the data and information of the stock
        df = stock_data[stock]
        stock_info = stock_info_data[stock]

        # Preprocess stock data
        # Current closing price
        current_close = df["Close"].iloc[-1]

        # Calculate the moving averages
        try:
            SMA_5 = df["SMA 5"].iloc[-1]
            SMA_20 = df["SMA 20"].iloc[-1]
            SMA_50 = df["SMA 50"].iloc[-1]
            SMA_200 = df["SMA 200"].iloc[-1]
            SMA_50_slope = df["SMA 50"].rolling(window=5).apply(slope_reg).iloc[-1]
            SMA_200_slope = df["SMA 200"].rolling(window=5).apply(slope_reg).iloc[-1]
        except Exception:
            EMA_5 = df["EMA 5"].iloc[-1]
            EMA_20 = df["EMA 20"].iloc[-1]
            EMA_50 = df["EMA 50"].iloc[-1]
            EMA_200 = df["EMA 200"].iloc[-1]
            EMA_50_slope = df["EMA 50"].rolling(window=5).apply(slope_reg).iloc[-1]
            EMA_200_slope = df["EMA 200"].rolling(window=5).apply(slope_reg).iloc[-1]

        # 52 week Low
        Low = round(min(df["Low"][-252:]), 2)

        # 52 week High
        High = round(max(df["High"][-252:]), 2)

        # RS rating and volume SMA 5 rank
        RS_rating, volume_sma5_rank = get_rs_volume(stock, rs_volume_df)

        # Check the Minervini conditions
        # Technicals
        try:
            cond_1 = current_close > SMA_50 > SMA_200
        except Exception:
            cond_1 = current_close > EMA_50 > EMA_200
        try:
            cond_2 = SMA_50_slope > 0
        except Exception:
            cond_2 = EMA_50_slope > 0
        try:
            cond_3 = SMA_200_slope > 0
        except Exception:
            cond_3 = EMA_200_slope > 0
        cond_4 = current_close >= (1.25 * Low)
        cond_5 = current_close >= (0.75 * High)
        conds_tech = cond_1 and cond_2 and cond_3 and cond_4 and cond_5

        # Preprocess stock information
        if conds_tech:
            if not backtest:
                market_cap = stock_info.get("marketCap", "N/A")
                market_cap = round(market_cap / 1e9, 2) if market_cap != "N/A" else "N/A"
                tEPS = stock_info.get("trailingEps", "N/A")
                fEPS = stock_info.get("forwardEps", "N/A")
                
                # Estimate the EPS growth of next year
                EPS_nextY_growth = round((fEPS - tEPS) / np.abs(tEPS) * 100, 2) if tEPS != "N/A" else "N/A"
            elif backtest:
                market_cap, EPS_past5Y_growth, EPS_thisY_growth, EPS_QoQ_growth, ROE = get_fundamentals(stock, index_name, end_date, current_date)
            sector = stock_info.get("sector", "N/A")
            industry = stock_info.get("industry", "N/A")

            # Fundamentals
            cond_6 = market_cap != "N/A" and market_cap > 1

            # Check if the conditions are met
            conds = conds_tech and cond_6
            if conds:
                if not backtest:
                    if index_name == "^HSI":
                        EPS_nextY_growth, earnings_thisQ_growth, ROE = get_fundamentals(stock, index_name, end_date, current_date)
                    else:
                        _, EPS_past5Y_growth, EPS_thisY_growth, EPS_QoQ_growth, ROE = get_fundamentals(stock, index_name, end_date, current_date)

                try:
                    cond_7 = EPS_thisY_growth >= 0
                except Exception:
                    cond_7 = False
                try:
                    cond_8 = EPS_QoQ_growth >= 10
                except Exception:
                    cond_8 = False
                try:
                    cond_9 = ROE >= 0
                except Exception:
                    cond_9 = False
                
                if cond_7 and cond_8 and cond_9:
                    # Get the quarterly growths of the stock
                    EPS_thisQ_growth, EPS_last1Q_growth, EPS_last2Q_growth = get_lastQ_growths(stock, index_name, end_date, current_date)

                    # Calculate the volatility of the stock over past 1 month
                    data = get_volatility(df)
                    volatility = data["Volatility"].iloc[-1]

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
                        "Volatility (%)": round(volatility * 100, 2),
                        "MA 5": SMA_5 if SMA_5 is not None else EMA_5,
                        "MA 20": SMA_20 if SMA_20 is not None else EMA_20,
                        "MA 50": SMA_50 if SMA_50 is not None else EMA_50,
                        "MA 200": SMA_200 if SMA_200 is not None else EMA_200,
                        "MA 5/20 Ratio": round(SMA_5 / SMA_20, 2) if SMA_5 is not None and SMA_20 is not None else round(EMA_5 / EMA_20, 2),
                        "MA 5/50 Ratio": round(SMA_5 / SMA_50, 2) if SMA_5 is not None and SMA_50 is not None else round(EMA_5 / EMA_50, 2),
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
                        "EPS this Y (%)": EPS_thisY_growth if index_name != "^HSI" else EPS_nextY_growth,
                        "EPS Q/Q (%)": EPS_QoQ_growth,
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
def EM_rating(data, factors, target_columns=["MVP Rating", "EPS this Y (%)", "EPS Q/Q (%)"]):
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
                  period, RS, NASDAQ_all, factors, backtest):
    # Define the path for the "Result" folder
    result_folder = "Result"

    # Get the infix
    infix = get_infix(index_name, index_dict, NASDAQ_all)

    # Iterate over all end dates
    for end_date in end_dates.copy():
        # Format the end date
        end_date_fmt = dt.datetime.strptime(end_date, "%Y-%m-%d").strftime("%d-%m-%y")

        # Define the folder path
        folder_path = os.path.join(result_folder, end_date_fmt)

        # Define the filename
        filename = os.path.join(folder_path, f"{infix}stock_{end_date_fmt}period{period}RS{RS}.xlsx")

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
        # Initialize two empty dictionaries to store the return multiples and volume SMAs
        return_muls = {}
        volume_smas = {}

        # Iterate over all stocks
        for ticker in tqdm(tickers):
            try:
                # Get the price data of the stock
                df = get_df(ticker, current_date)

                # Filter the data
                df = df[df.index <= end_date]

                # Calculate the percent change of the stock
                df["Percent Change"] = df["Close"].pct_change()

                # Calculate the stock return
                stock_return = (df["Percent Change"] + 1).tail(period).cumprod().iloc[-1]

                # Calculate the stock return relative to the market
                return_mul = round((stock_return / index_return), 2)
                return_muls[ticker] = return_mul
                print(f"Ticker: {ticker} ; Return multiple against {index_shortName}: {return_mul}\n")

                # Calculate the moving averages of volume
                df["Volume SMA 5"] = SMA(df, 5, column="Volume")
                df["Volume SMA 20"] = SMA(df, 20, column="Volume")
                volume_smas[ticker] = {"Volume SMA 5": df["Volume SMA 5"].iloc[-1], "Volume SMA 20": df["Volume SMA 20"].iloc[-1]}
            except Exception as e:
                print(f"Error gathering data for {ticker}: {e}\n")
                
                continue
            # time.sleep(0.05)

        # Create a dataframe to store the RS ratings of tickers
        return_muls = dict(sorted(return_muls.items(), key=lambda x: x[1], reverse=True))
        rs_df = pd.DataFrame(return_muls.items(), columns=["Ticker", "Value"])
        rs_df["RS"] = rs_df["Value"].rank(pct=True) * 100
        rs_df = rs_df[["Ticker", "RS"]]

        # Create a dataframe to store the volumes of tickers
        volume_df = pd.DataFrame.from_dict(volume_smas, orient="index", columns=["Volume SMA 5", "Volume SMA 20"])
        volume_df["Ticker"] = volume_df.index
        volume_df.reset_index(drop=True, inplace=True)
        volume_df["Volume SMA 5 Rank"] = volume_df["Volume SMA 5"].rank(ascending=False)
        volume_df["Volume SMA 20 Rank"] = volume_df["Volume SMA 20"].rank(ascending=False)

        # Merge the dataframes
        rs_volume_df = pd.merge(rs_df, volume_df, on="Ticker")
        rs_volume_df = rs_volume_df.sort_values(by="RS", ascending=False)

        # Save the merged dataframe to a .csv file
        filename = os.path.join(result_folder, f"{infix}rs_volume.csv")
        if not backtest:
            rs_volume_df.to_csv(filename, index=False)

        # Filter the RS dataframe
        rs_df = rs_df[rs_df["RS"] >= RS]

        # Fetch the stock data and stock information in parallel
        stocks = rs_df["Ticker"]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            stock_data = {stock: data for stock, data in zip(stocks, executor.map(lambda stock: get_stock_data(stock, end_date, current_date), stocks))}
            stock_info_data = {stock: info for stock, info in zip(stocks, executor.map(get_stock_info, stocks))}

        # Process each stock and create an export list
        export_data = [process_stock(stock, index_name, end_date, current_date, stock_data, stock_info_data, rs_volume_df, backtest=backtest) for stock in tqdm(stocks)]
        export_data = [row for row in export_data if row is not None]
        export_list = pd.DataFrame(export_data)
        if index_name == "^HSI":
            pass
        else:
            export_list = EM_rating(export_list, factors)

        # Format the end date
        end_date_fmt = dt.datetime.strptime(end_date, "%Y-%m-%d").strftime("%d-%m-%y")

        # Check if the "end_date_fmt" folder exists inside the "Result" folder, create it if it does not
        folder_path = os.path.join(result_folder, f"{end_date_fmt}")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Export the results to an Excel file inside the "end_date_fmt" folder
        filename = os.path.join(folder_path, f"{infix}stock_{end_date_fmt}period{period}RS{RS}.xlsx")
        writer = EW(filename)
        export_list.to_excel(writer, "Sheet1")
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

    # Initial setup
    current_date = start.strftime("%Y-%m-%d")

    # Define the paths for the folders
    folders = ["Price data"]
    
    # Check if the folders exist, create them if they do not
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # Create the end dates
    end_dates = generate_end_dates(5, current_date)
    end_dates.append(current_date)
    end_dates = [current_date]

    # Variables
    NASDAQ_all = True
    period = 252
    RS = 90
    factors = [1, 1, 1]
    backtest = False

    # Index
    index_name = "^GSPC"
    index_dict = {"^HSI": "HKEX", "^GSPC": "S&P 500", "^IXIC": "NASDAQ Composite"}

    # Stock selection
    select_stocks(end_dates, current_date, index_name, index_dict, 
                  period, RS, NASDAQ_all, factors, backtest)

    # Print the end time and total runtime
    end = dt.datetime.now()
    print(end, "\n")
    print("The program used", end - start)

# Run the main function
if __name__ == "__main__":
    main()