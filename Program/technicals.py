# Imports
from helper_functions import get_df, slope_reg
import numpy as np
import pandas as pd
from tqdm import tqdm

# Calculate the simple moving average (SMA)
def SMA(data, period, column="Close"):
    return data[column].rolling(window=period).mean()

# Calculate the exponential moving average (EMA)
def EMA(data, period, column="Close"):
    return data[column].ewm(span=period, adjust=False).mean()

# Get the volatility
def get_volatility(data, periods=[20, 60], column="Close"):
    data_copy = data.copy()

    # Calculate the percent change of the stock
    data_copy["Percent Change"] = data_copy[column].pct_change()

    # Calculate the volatility
    for period in periods:
        data[f"Volatility {period}"] = data_copy["Percent Change"].rolling(window=period).std()

    return data

# Calculate the average true range (ATR)
def ATR(data, period=14, column="Close"):
    # Calculate the true range (TR)
    TR = pd.concat([
        abs(data["High"] - data["Low"]),
        abs(data["High"] - data[column].shift()),
        abs(data["Low"] - data[column].shift())
        ], axis=1).max(axis=1)
    
    # Calculate the ATR by EMA of TR
    ATR = TR.ewm(span=period, adjust=False).mean()
    data["ATR"] = ATR

    return data

# Calculate the moving average convergence/divergence (MACD)
def MACD(data, period_long, period_short, period_signal, column="Close"):
    # Calculate the short EMA
    EMA_short = EMA(data, period_short, column=column)

    # Calculate the long EMA
    EMA_long = EMA(data, period_long, column=column)

    # Calculate the MACD
    data["MACD"] = EMA_short - EMA_long

    # Calculate the signal line
    data["Signal Line"] = EMA(data, period_signal, column="MACD")
    
    return data

# Calculate the Relative Strength Index (RSI)
def RSI(data, period=14, column="Close"):
    # Calculate the change of the stock
    data["Change"] = data[column].diff()

    # Calculate the gains and losses
    gain = data["Change"].copy()
    loss = data["Change"].copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0

    # Calculate the relative strength (RS)
    RS = gain.rolling(window=period).mean() / abs(loss.rolling(window=period).mean())

    # Calculate the RSI
    RSI = 100 - (100 / (1 + RS))
    data["RSI"] = RSI

    return data

# Calculate the Relative Momentum Index (RMI)
def RMI(data, period=20, momentum=3, column="Close"):
    data_copy = data.copy()

    # Calculate the change of the stock
    data_copy["Change"] = data_copy[column].diff(momentum)[momentum:]

    # Calculate the gains and losses
    gain = data_copy["Change"].copy()
    loss = data_copy["Change"].copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0

    # Calculate the relative momentum (RM)
    RM = gain.rolling(window=period).mean() / abs(loss.rolling(window=period).mean())

    # Calculate the RMI
    RMI = 100 - (100 / (1 + RM))
    data["RMI"] = RMI

    return data

# Calculate the Money Flow Index (MFI)
def MFI(data, period=14, column="Close"):
    data_copy = data.copy()

    # Calculate HLC3, Raw MF, and the change of HLC3
    data_copy["HLC3"] = (data_copy["High"] + data_copy["Low"] + data_copy["Close"]) / 3
    data_copy["Raw MF"] = data_copy["HLC3"] * data_copy["Volume"]
    data_copy["HLC3 Change"] = data_copy["HLC3"].diff()

    # Calculate the +MF and -MF
    data_copy["+MF"] = np.where(data_copy["HLC3 Change"] > 0, data_copy["Raw MF"], 0)
    data_copy["-MF"] = np.where(data_copy["HLC3 Change"] < 0, data_copy["Raw MF"], 0)

    # Calculate the sum of +MF and -MF over a period
    data_copy["+MF Sum"] = data_copy["+MF"].rolling(window=period).sum()
    data_copy["-MF Sum"] = data_copy["-MF"].rolling(window=period).sum()

    # Calculate the MF ratio
    data_copy["MF Ratio"] = data_copy["+MF Sum"] / abs(data_copy["-MF Sum"])

    # Calcualte the MFI
    data["MFI"] = 100 - (100 / (1 + data_copy["MF Ratio"]))

    return data

# Calculate the Commodity Channel Index (CCI)
def CCI(data, period=20):
    data_copy = data.copy()
    
    # Calculate the average of high, low and closing prices (HLC3)
    data_copy["HLC3"] = (data_copy["High"] + data_copy["Low"] + data_copy["Close"]) / 3

    # Calculate the moving average of HLC3
    data_copy["MA"] = data_copy["HLC3"].rolling(window=period).mean()

    # Calculate the CCI
    data["CCI"] = (data_copy["HLC3"] - data_copy["MA"]) / (0.015 * data_copy["HLC3"].rolling(window=period).std())

    return data

# Calculate the Average Directional Index (ADX)
def ADX(data, period=40, column="Close"):
    data_copy = data.copy()

    # Calculate the change of the stock
    data_copy["Change"] = data_copy[column].diff()

    # Calculate the +DI and -DI
    data_copy["+DI"] = np.where(data_copy["Change"] > 0, data_copy["Change"], 0)
    data_copy["-DI"] = np.where(data_copy["Change"] < 0, abs(data_copy["Change"]), 0)

    # Calculate the EMA of +DI and -DI
    data_copy["+DI EMA"] = EMA(data_copy, period, column="+DI")
    data_copy["-DI EMA"] = EMA(data_copy, period, column="-DI")

    # Calculate the percentage difference between the mean of +DI and -DI
    data_copy["DI Percent Difference"] = np.abs(data_copy["+DI EMA"] - data_copy["-DI EMA"]) / (data_copy["+DI EMA"] + data_copy["-DI EMA"]) * 100

    # Calculate the ADX
    data["ADX"] = data_copy["DI Percent Difference"].rolling(window=period).mean()

    return data

# Calcualte the OB/OS indicator (OBOS)
def OBOS(data, period=14, column="Close"):
    data_copy = data.copy()

    # Calculate the highest and lowest closing price over the past period
    data_copy["HC"] = data_copy[column].rolling(window=period).max()
    data_copy["LC"] = data_copy[column].rolling(window=period).min()

    # Calculate the OB/OS indicator
    data["OBOS"] = (data_copy["Close"] - data_copy["LC"]) / (data_copy["HC"] - data_copy["LC"]) * 100

    return data

# Calculate the bull/bear power
def bull_bear(data, period=13, column="Close"):
    data_copy = data.copy()

    # Calculate the bull power
    data["Bull"] = data_copy["High"] - EMA(data_copy, period, column=column)

    # Calculate the bear power
    data["Bear"] = data_copy["Low"] - EMA(data_copy, period, column=column)

    # Calculate the total bull/bear power
    data["Bull Bear"] = data["Bull"] + data["Bear"]

    return data

# Calculate the MVP/VCP indicator
def MVP_VCP(data, period_MVP=15, period_VCP=10, contraction=0.05, period=60, column="Close"):
    data_copy = data.copy()
    
    # Check if the M, V, and P conditions are met
    data_copy["M"] = data_copy["Close"].diff().rolling(window=period_MVP).apply(lambda x: (x > 0).sum()).ge(int(period_MVP * 0.8))
    data_copy["V"] = (data_copy["Volume"] >= data_copy["Volume"].shift(period_MVP) * 1.2)
    data_copy["P"] = (data_copy["Close"] >= data_copy["Close"].shift(period_MVP) * 1.2)
    data_copy["MVP"] = ""
    data_copy.loc[data_copy["M"] & ~data_copy["V"] & ~data_copy["P"], "MVP"] = "M"
    data_copy.loc[data_copy["M"] & data_copy["V"] & ~data_copy["P"], "MVP"] = "MV"
    data_copy.loc[data_copy["M"] & data_copy["P"] & ~data_copy["V"], "MVP"] = "MP"
    data_copy.loc[data_copy["M"] & data_copy["V"] & data_copy["P"], "MVP"] = "MVP"
    data["MVP"] = data_copy["MVP"]

    # Count the number of occurrences of M, V, and P over the past period
    data[f"M past {period}"] = data_copy["MVP"].apply(lambda x: x == "M").rolling(window=period).sum()
    data[f"MV past {period}"] = data_copy["MVP"].apply(lambda x: x == "MV").rolling(window=period).sum()
    data[f"MP past {period}"] = data_copy["MVP"].apply(lambda x: x == "MP").rolling(window=period).sum()
    data[f"MVP past {period}"] = data_copy["MVP"].apply(lambda x: x == "MVP").rolling(window=period).sum()

    # Calculate the MVP ratng
    data["MVP Rating"] = ((1 / 3 * data[f"M past {period}"]) + (2 / 3 * (data[f"MV past {period}"] + data[f"MP past {period}"])) + data[f"MVP past {period}"]) / 60 * 100

    # Calculate the highest, median, and lowest closing price over the past period
    data_copy["HC"] = data_copy[column].rolling(window=period_VCP).max()
    data_copy["MC"] = data_copy[column].rolling(window=period_VCP).median()
    data_copy["LC"] = data_copy[column].rolling(window=period_VCP).min()

    # Check if the highest and lowest closing prices differ by less than contraction
    data["VCP"] = (1 - data_copy["LC"] / data_copy["HC"]) <= contraction

    # Check if pivot breakout occurs
    data["Pivot breakout"] = data_copy[column] > 1 / 3 * (data_copy["HC"] + data_copy["MC"] + data_copy["LC"])

    # Check if the volume is shrinking
    data["Volume shrinking"] = data_copy["Volume"].rolling(window=period_VCP).apply(slope_reg) < 0
    
    return data

# Check follow-through day (FTD) and distribution day (DD)
def FTD_DD(data, period=50, threshold=0.015, column="Close"):
    # Check FTD
    data["FTD"] = (data[column] > (1 + threshold) * data[column].shift(1)) \
        & (data["Volume"] > data["Volume"].shift(1)) \
        & (data["Volume"] > data["Volume"].rolling(window=period).mean())
    
    # Check DD
    data["DD"] = (data[column] < (1 - threshold) * data[column].shift(1)) \
    & (data["Volume"] > data["Volume"].shift(1)) \
    & (data["Volume"] > data["Volume"].rolling(window=period).mean())

    return data

# Check if there are at least four FTDs or DDs recently
def multiple_FTD_DD(data, period=10, columns=["FTD", "DD"]):
    data["Multiple FTDs"] = data[columns[0]].rolling(period).sum() >= 4
    data["Multiple DDs"] = data[columns[1]].rolling(period).sum() >= 4

    return data

# Calculate the Z-Score
def calculate_ZScore(data, indicators, period):
    for indicator in indicators:
        # Calculat the mean of indicator
        data[f"{indicator} Mean"] = data[f"{indicator}"].rolling(window=period).mean()

        # Calculate the SD of indicator
        data[f"{indicator} SD"] = data[f"{indicator}"].rolling(window=period).std()

        # Calculate the z-score of indicator
        data[f"{indicator} Z-Score"] = (data[f"{indicator}"] - data[f"{indicator} Mean"]) / data[f"{indicator} SD"]

    return data

# Add technical indicators to the data
def add_indicator(data):
    get_volatility(data)
    ATR(data)
    MACD(data, 26, 12, 9)
    RSI(data)
    RMI(data)
    MFI(data)
    CCI(data)
    ADX(data)
    OBOS(data)
    bull_bear(data)
    MVP_VCP(data)
    FTD_DD(data)
    multiple_FTD_DD(data)

    # Calculate the moving averages of closing prices and volumes
    periods = [5, 20, 50, 200]
    for i in periods:
        data[f"SMA {str(i)}"] = SMA(data, i)
        data[f"EMA {str(i)}"] = EMA(data, i)
        data[f"Volume SMA {str(i)}"] = SMA(data, i, column="Volume")

    return data

# Preprocess the data to get the market breadth and AD line
def trend_AD(data, periods=[20, 50, 200], column="Close"):
    data_copy = data.copy()

    # Calculate the SMAs
    for i in periods:
        data_copy[f"SMA {str(i)}"] = SMA(data_copy, i, column=column)

        # Check if the closing price is above SMAs
        data[f"Above SMA {str(i)}"] = 0
        data.loc[data_copy[column] > data_copy[f"SMA {str(i)}"], f"Above SMA {str(i)}"] = 1
        data.loc[data_copy[column] <= data_copy[f"SMA {str(i)}"], f"Above SMA {str(i)}"] = 0

    # Calculate the change of the stock
    data_copy["Change"] = data_copy[column].diff()

    # Initialize the advancing (A) and declining (D) columns
    data["A"] = 0
    data["D"] = 0

    # Check if the price advances (A) or declines (D)
    data.loc[data_copy["Change"] > 0, "A"] = 1
    data.loc[data_copy["Change"] <= 0, "D"] = 1

    return data

# Calculate the market breadth indicators
def market_breadth(end_date, index_df, tickers, periods=[20, 50, 200]):
    # Initialize the Above SMA columns
    for i in periods:
        index_df[f"Above SMA {str(i)}"] = 0

    # Initialize the advancing (A) and declining (D) columns
    index_df["A"] = 0
    index_df["D"] = 0

    # Iterate over all tickers
    for ticker in tqdm(tickers):
        # Get the price data of the ticker
        df = get_df(ticker, end_date)

        # Check if the data exist
        if df is not None:
            # Preprocess the data to get the market breadth and AD line
            df = trend_AD(df)

            # Calculate the number of tickers above SMAs
            for i in periods:
                index_df.loc[:, f"Above SMA {str(i)}"] = index_df.loc[:, f"Above SMA {str(i)}"].add(df[f"Above SMA {str(i)}"], fill_value=0)
            
            # Accumulate the advancing (A) and declining (D) values for all tickers
            index_df.loc[:, "A"] = index_df.loc[:, "A"].add(df["A"], fill_value=0)
            index_df.loc[:, "D"] = index_df.loc[:, "D"].add(df["D"], fill_value=0)

    # Calculate the AD line
    index_df["AD Change"] = index_df["A"] - index_df["D"]
    index_df["AD"] = index_df["AD Change"].cumsum()
    
    return index_df

# Calculate the JdK RS-Ratio and Momentum
def get_JdK(sectors, index_df, end_date, period_short=12, period_long=26, period_signal=9):
    # Iterate over all sectors
    for sector in tqdm(sectors):
        # Get the price data of the sector
        df = get_df(sector, end_date)
        df_copy = df.copy()

        # Calculate the closing price relative to benchmark
        df_copy["Relative Close"] = df["Close"] / index_df["Close"]

        # Calculate the SMAs of relative closing price
        df_copy[f"Relative Close SMA {period_short}"] = df_copy["Relative Close"].rolling(window=period_short).mean()
        df_copy[f"Relative Close SMA {period_long}"] = df_copy["Relative Close"].rolling(window=period_long).mean()

        # Calculate the JdK RS-Ratio
        df_copy["JdK RS-Ratio"] = 100 * ((df_copy[f"Relative Close SMA {period_short}"] - df_copy[f"Relative Close SMA {period_long}"]) / df_copy[f"Relative Close SMA {period_long}"] + 1)

        # Calculate the SMA of JdK RS-Ratio
        df_copy[f"JdK RS-Ratio SMA {period_signal}"] = df_copy["JdK RS-Ratio"].rolling(window=period_signal).mean()

        # Calculate the JdK RS-Momentum
        df_copy["JdK RS-Momentum"] = 100 * ((df_copy["JdK RS-Ratio"] - df_copy[f"JdK RS-Ratio SMA {period_signal}"]) / df_copy[f"JdK RS-Ratio SMA {period_signal}"] + 1)

        # Insert the results into index_df
        index_df[f"{sector} JdK RS-Ratio"] = df_copy["JdK RS-Ratio"]
        index_df[f"{sector} JdK RS-Momentum"] = df_copy["JdK RS-Momentum"]

        # Fill NaN values with the previous value
        index_df = index_df.ffill()

    return index_df