# Imports
import datetime as dt
from dateutil.relativedelta import relativedelta
from helper_functions import get_df, get_earning_dates
import json
import numpy as np
import os
import pandas as pd
from pyfinviz.quote import Quote
import re
import requests
import time
from tqdm import tqdm
import yfinance as yf

# Scrape the url
def scrape(url, retry=0, **kwargs):
    session = requests.Session()

    # Add the headers to mimic a browser
    session.headers.update(
        {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
         "Accept-Language": "en-US,en;q=0.9",
         "Referer": "https://github.com/ckkyue?tab=repositories",
         "Cache-Control": "no-cache",
         "DNT": "1",
         }
    )

    try:
        response = session.get(url, **kwargs)
        response.raise_for_status() # Raise an exception for non-2xx status codes
        return response

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            if retry < 10: # Maximum retry attempts set to 10
                retry_after = np.ceil(float(e.response.headers.get("Retry-After", 20))) # Default to 20 seconds if Retry-After header is missing
                time.sleep(retry_after)
                return scrape(url, retry=retry+1, **kwargs) # Retry the scrape after waiting
            else:
                return f"Maximum number of retry attempts reached."
        else:
            return f"HTTP error: {e}"

    except Exception as e:
        return f"General error: {e}"

# Create the dataframe
def etl(response):
    try:
        # Regex to find the data
        num = re.findall('(?<=div\>\"\,)[0-9\.\"\:\-\, ]*', response.text)
        text = re.findall('(?<=s\: \')\S+(?=\'\, freq)', response.text)

        # Convert the text to a dictionary
        dicts = [json.loads("{" + i + "}") for i in num]

        # Initialize an empty dataframe to store the data
        df = pd.DataFrame()
        for index, value in enumerate(text):
            df[value] = dicts[index].values()
        df.index = dicts[index].keys()

        return df
    except Exception:
        return pd.DataFrame()

# Get the fundamentals of a stock and save the data to a .csv file
def fundamentals_csv(stock, end_date):
    # Define the folder path
    folder_path = "Fundamentals"

    # Check if there are pre-existing data
    current_files = [file for file in os.listdir(folder_path) if file.startswith(f"{stock}_fundamentals_")]

    # Get the list of dates
    dates = [file.split("_")[-1].replace(".csv", "") for file in current_files]

    # Get the maximum date from the list of dates
    max_date = max(dates) if dates else "N/A"

    # Remove the old files for dates prior to the maximum date
    if max_date != "N/A":
        for date in dates:
            if date < max_date:
                os.remove(os.path.join(folder_path, f"{stock}_fundamentals_{date}.csv"))
        # Define the filename
        if end_date >= max_date:
            filename = os.path.join(folder_path, f"{stock}_fundamentals_{end_date}.csv")
        else:
            filename = os.path.join(folder_path, f"{stock}_fundamentals_{max_date}.csv")
    else:
        filename = os.path.join(folder_path, f"{stock}_fundamentals_{end_date}.csv")
    
    # Save the data to a .csv file if the most updated data do not exist
    if not os.path.isfile(filename):
        try:
            stock_upper = stock.upper()
            url_revenue = f"https://www.macrotrends.net/stocks/charts/{stock_upper}/unknown/revenue"
            response_revenue = scrape(url_revenue)

            try:
                base_url = response_revenue.url

            except AttributeError:
                base_url = None
                print(f"AttributeError: response_revenue = {response_revenue}")

                return None

            # Get the infix
            infix = base_url.split("/")[-2]

            # Set the urls for scraping
            url1 = f"https://www.macrotrends.net/stocks/charts/{stock_upper}/{infix}/income-statement?freq=Q"
            url2 = f"https://www.macrotrends.net/stocks/charts/{stock_upper}/{infix}/financial-ratios?freq=Q"

            # Scrape the urls and create dataframes
            response1 = scrape(url1)
            df1 = etl(response1)
            response2 = scrape(url2)
            df2 = etl(response2)

            # Check if df1 or df2 is empty
            if df1.empty or df2.empty:
                print(f"Empty dataframe for {stock}.")

                return None

            df = pd.concat([df1, df2], axis=1) # Concatenate based on index
            df.index.name = "Date"
            df.to_csv(filename)
            print(f"Fundamentals data download completed for {stock}.")
            
            if max_date != "N/A":
                os.remove(os.path.join(folder_path, f"{stock}_fundamentals_{max_date}.csv"))
        
        except Exception as e:
            print(f"Error for {stock}: {e}")
            print(f"Retrieve existing data at {max_date}.")
            filename = os.path.join(folder_path, f"{stock}_fundamentals_{max_date}.csv")
    else:
        print(f"Fundamentals data download completed for {stock} before.")

    # Read the fundamentals data of the stock
    df = pd.read_csv(filename)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    return df

# Define the fundamentals map
def fundamentals_map(x):
    try:
        return float(x.replace("%", ""))
    except ValueError:
        return "N/A"

# Get the market cap of a ticker
def get_market_cap(ticker, stock_info, end_date, current_date):
    try:
        # Get the earning dates
        earning_dates = get_earning_dates(ticker)
        earning_dates = earning_dates[earning_dates < current_date]

    except Exception as e:
        print(f"Error getting earnings dates {ticker}: {e}\n")
        earning_dates = None

    # Get the most recent earning date
    if earning_dates is not None:
        recent_earning_date = earning_dates.max().strftime("%Y-%m-%d")
    else:
        recent_earning_date = current_date

    # Read the fundamentals data from .csv file if end date is earlier than recent earning date
    if end_date < recent_earning_date:
        if earning_dates is not None:
            # Get the earning dates
            earning_dates = earning_dates[earning_dates < end_date]
            
            # Get the most recent report date
            recent_report_date = earning_dates.max().strftime("%Y-%m-%d")

        # Estimate the recent report date by shifting 3 months backwards from end date
        else:
            recent_report_date = (dt.datetime.strptime(end_date, "%Y-%m-%d") - relativedelta(months=3)).strftime("%Y-%m-%d")

        # Define the csv date
        csv_date = "2024-12-01"

        # Read the fundamentals data from .csv file
        df = fundamentals_csv(ticker, csv_date)

        # Check if the dataframe is None
        if df is not None:
            # Filter the dates
            dates = df.index[df.index <= recent_report_date]

            # Generate the dates
            closest_date = dates[0]

            # Get the number of shares
            shares = df.loc[closest_date, "shares-outstanding"] * 1e6

            # Get the price data of the stock
            df = get_df(ticker, end_date)

            # Get the latest closing price
            closest_close = df.loc[df.index[df.index <= closest_date].max(), "Close"]

            # Calculate the market capitalization (market cap)
            market_cap = round(shares * closest_close / 1e9, 2)
        
        else:
            market_cap = "N/A"
        
    else:
        market_cap = stock_info.get("marketCap", "N/A")
        market_cap = round(market_cap / 1e9, 2) if market_cap != "N/A" else "N/A"

    return market_cap

# Get the fundamentals data of a ticker
def get_fundamentals(ticker, end_date, current_date, columns=["EPS past 5Y", "EPS this Y", "EPS Q/Q", "ROE"]):
    try:
        # Get the earning dates
        earning_dates = get_earning_dates(ticker)
        earning_dates = earning_dates[earning_dates < current_date]

    except Exception as e:
        print(f"Error getting earnings dates {ticker}: {e}\n")
        earning_dates = None

    # Get the most recent earning date
    if earning_dates is not None:
        recent_earning_date = earning_dates.max().strftime("%Y-%m-%d")

    else:
        recent_earning_date = current_date
    
    # Read the fundamentals data from .csv file if end date is earlier than recent earning date
    if end_date < recent_earning_date:
        if earning_dates is not None:
            # Get the earning dates
            earning_dates = earning_dates[earning_dates < end_date]
            
            # Get the most recent report date
            recent_report_date = earning_dates.max().strftime("%Y-%m-%d")

        # Estimate the recent report date by shifting 3 months backwards from end date
        else:
            recent_report_date = (dt.datetime.strptime(end_date, "%Y-%m-%d") - relativedelta(months=3)).strftime("%Y-%m-%d")

        # Define the csv date
        csv_date = "2024-12-01"

        # Read the fundamentals data from .csv file
        df = fundamentals_csv(ticker, csv_date)

        # Check if the dataframe is None
        if df is not None:
            # Filter the dates
            dates = df.index[df.index <= recent_report_date]

            # Generate the dates
            closest_date = dates[0]
            past_dates_annual = {str(i): dates[i * 4 - 1] if i * 4 - 1 < len(dates) else None for i in [1, 2, 5, 6]}
            past_dates_quarter = {str(i): dates[i] if i < len(dates) else None for i in [1, 2, 3]}
            
            # Get the earnings per share (EPS) values
            EPS_thisQ = df.loc[closest_date, "eps-basic-net-earnings-per-share"]
            EPS_last3Q = df.loc[past_dates_quarter["3"], "eps-basic-net-earnings-per-share"] if past_dates_quarter["3"] is not None else "N/A"

            try:
                EPS_thisY = df.loc[closest_date : past_dates_annual["1"], "eps-basic-net-earnings-per-share"].sum()
            except Exception:
                EPS_thisY = "N/A"
            try:
                EPS_last5Y = df.loc[past_dates_annual["5"] : past_dates_annual["6"], "eps-basic-net-earnings-per-share"].sum()
            except Exception:
                EPS_last5Y = "N/A"
            try:
                EPS_lastY = df.loc[past_dates_annual["1"] : past_dates_annual["2"], "eps-basic-net-earnings-per-share"].sum()
            except Exception:
                EPS_lastY = "N/A"

            # Calculate the growth rates of EPS values
            EPS_past5Y_growth = round(1 / 5 * (EPS_thisY / EPS_last5Y) / np.abs(EPS_last5Y) * 100, 2) if EPS_thisY != "N/A" and EPS_last5Y != "N/A" and EPS_last5Y != 0 else "N/A"
            EPS_thisY_growth = round((EPS_thisY - EPS_lastY) / np.abs(EPS_lastY) * 100, 2) if EPS_thisY != "N/A" and EPS_lastY != "N/A" and EPS_lastY != 0 else "N/A"
            EPS_QoQ_growth = round((EPS_thisQ - EPS_last3Q) / np.abs(EPS_last3Q) * 100, 2) if EPS_thisQ != "N/A" and EPS_last3Q != "N/A" and EPS_last3Q != 0 else "N/A"
            
            # Get the ROE
            ROE = round(df.loc[closest_date, "roe"], 2)

        else:
            EPS_past5Y_growth, EPS_thisY_growth, EPS_QoQ_growth, ROE = "N/A", "N/A", "N/A", "N/A"

    # Scrape the fundamentals data from Finviz if end date is later than recent earning date
    else:
        quote = Quote(ticker=ticker)
        fundamental_df = quote.fundamental_df.loc[:, columns].map(fundamentals_map)
        data = fundamental_df.values[0]
        EPS_past5Y_growth, EPS_thisY_growth, EPS_QoQ_growth, ROE = *data,
    
    return EPS_past5Y_growth, EPS_thisY_growth, EPS_QoQ_growth, ROE

# Get the quarterly growths of a ticker
def get_lastQ_growths(ticker, index_name, end_date, current_date):
    try:
        # Get the earning dates
        earning_dates = get_earning_dates(ticker)
        earning_dates = earning_dates[earning_dates < current_date]
    except Exception as e:
        print(f"Error getting earnings dates {ticker}: {e}\n")
        earning_dates = None

    # Return "N/A" for HKEX stocks
    if index_name == "^HSI":
        EPS_thisQ_growth, EPS_last1Q_growth, EPS_last2Q_growth = "N/A", "N/A", "N/A"
    
    # Handle non-HKEX stocks
    else:
        if earning_dates is not None:
            # Get the earning dates
            earning_dates = earning_dates[earning_dates < end_date]
            
            # Get the most recent report date
            recent_report_date = earning_dates.max().strftime("%Y-%m-%d")

        # Estimate the recent report date by shifting 3 months backward from end date
        else:
            recent_report_date = (dt.datetime.strptime(end_date, "%Y-%m-%d") - relativedelta(months=3)).strftime("%Y-%m-%d")

        # Define the csv date
        csv_date = "2024-12-01"

        # Modify the csv date
        if recent_report_date > csv_date:
            csv_date = recent_report_date
        
         # Read the fundamentals data from .csv file       
        df = fundamentals_csv(ticker, csv_date)

        # Check if the dataframe is None
        if df is not None:
            # Filter the dates
            dates = df.index[df.index <= recent_report_date]

            # Generate the dates
            closest_date = dates[0]
            past_dates_quarter = {str(i): dates[i] if i < len(dates) else None for i in [1, 2, 3]}
            
            # Get the EPS values
            EPS_thisQ = df.loc[closest_date, "eps-basic-net-earnings-per-share"]
            EPS_last1Q = df.loc[past_dates_quarter["1"], "eps-basic-net-earnings-per-share"] if past_dates_quarter["1"] is not None else "N/A"
            EPS_last2Q = df.loc[past_dates_quarter["2"], "eps-basic-net-earnings-per-share"] if past_dates_quarter["2"] is not None else "N/A"
            EPS_last3Q = df.loc[past_dates_quarter["3"], "eps-basic-net-earnings-per-share"] if past_dates_quarter["3"] is not None else "N/A"
            
            # Calculate the growth rates of EPS values
            EPS_thisQ_growth = round((EPS_thisQ - EPS_last1Q) / np.abs(EPS_last1Q) * 100, 2) if EPS_last1Q != "N/A" else "N/A"
            EPS_last1Q_growth = round((EPS_last1Q - EPS_last2Q) / np.abs(EPS_last2Q) * 100, 2) if EPS_last2Q != "N/A" else "N/A"
            EPS_last2Q_growth = round((EPS_last2Q - EPS_last3Q) / np.abs(EPS_last3Q) * 100, 2) if EPS_last3Q != "N/A" else "N/A"

        else:
            EPS_thisQ_growth, EPS_last1Q_growth, EPS_last2Q_growth = "N/A", "N/A", "N/A"

    return EPS_thisQ_growth, EPS_last1Q_growth, EPS_last2Q_growth