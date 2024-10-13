# Imports
import datetime as dt
from dateutil.relativedelta import relativedelta
from helper_functions import get_df, get_infix, merge_stocks
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.signal import argrelextrema
import seaborn as sns
from statsmodels.tsa.stattools import acf
from technicals import *

# Visualize the closing price history
def plot_close(stock, df, show=120, MVP_VCP=True, save=False):
    # Add technical indicators to the data
    add_indicator(df)

    # Filter the data
    df = df[- show:]

    # Define the widths
    width_candle = 1
    width_stick = 0.2

    # Separate the dataframe into green and red candlesticks
    up_df = df[df["Close"] >= df["Open"]]
    down_df = df[df["Close"] <= df["Open"]]
    colour_up = "green"
    colour_down = "red"

    # Create a figure with two subplots, one for the closing price and one for the volume
    if stock == "^VIX":
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [5, 1]}, sharex=True)
        
    # Plot the up prices on the top subplot
    ax1.bar(up_df.index, up_df["Close"] - up_df["Open"], width_candle, bottom=up_df["Open"], color=colour_up)
    ax1.bar(up_df.index, up_df["High"] - up_df["Close"], width_stick, bottom=up_df["Close"], color=colour_up)
    ax1.bar(up_df.index, up_df["Low"] - up_df["Open"], width_stick, bottom=up_df["Open"], color=colour_up)

    # Plot the down prices on the top subplot
    ax1.bar(down_df.index, down_df["Close"] - down_df["Open"], width_candle, bottom=down_df["Open"], color=colour_down)
    ax1.bar(down_df.index, down_df["High"] - down_df["Open"], width_stick, bottom=down_df["Open"], color=colour_down)
    ax1.bar(down_df.index, down_df["Low"] - down_df["Close"], width_stick, bottom=down_df["Close"], color=colour_down)

    # Plot the MVP and VCP conditions on the top subplot
    if MVP_VCP:
        ax1.scatter(df.index[df["MVP"] == "M"], df["Close"][df["MVP"] == "M"], marker="^", color="grey", label="M")
        ax1.scatter(df.index[df["MVP"] == "MP"], df["Close"][df["MVP"] == "MP"], marker="^", color="yellow", label="MP")
        ax1.scatter(df.index[df["MVP"] == "MV"], df["Close"][df["MVP"] == "MV"], marker="^", color="blue", label="MV")
        ax1.scatter(df.index[df["MVP"] == "MVP"], df["Close"][df["MVP"] == "MVP"], marker="^", color="green", label="MVP")
        ax1.scatter(df.index[df["VCP"] == True], df["Close"][df["VCP"] == True], marker=">", color="orange", label="VCP")
    else:
        pass

    # Plot the SMAs on the top subplot
    periods = [5, 20, 50, 200]
    for i in periods:
        ax1.plot(df[f"SMA {str(i)}"], label=f"SMA {str(i)}")

    # Set the y label of the top subplot
    ax1.set_ylabel("Price")

    # Set the x limit of the top subplot
    buffer = relativedelta(days=1)
    ax1.set_xlim(df.index[0] - buffer, df.index[-1] + buffer)

    if stock != "^VIX":
        # Plot the volume on the bottom subplot
        ax2.bar(up_df.index, up_df["Volume"], label="Volume (+)", color=colour_up)
        ax2.bar(down_df.index, down_df["Volume"], label="Volume (-)", color=colour_down)

        # Plot the volume SMA 50 on the bottom subplot
        ax2.plot(df["Volume SMA 50"], label="Volume SMA 50", color="purple")

        # Set the label of the bottom subplot
        ax2.set_ylabel("Volume")

        # Set the x label
        plt.xlabel("Date")

        # Combine the legends and place them at the top subplot
        handles, labels = ax1.get_legend_handles_labels()
        handles += ax2.get_legend_handles_labels()[0]
        labels += ax2.get_legend_handles_labels()[1]
        ax1.legend(handles, labels)

    # Set the title
    plt.suptitle(f"Closing price history for {stock}")

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Save the plot
    if save:
        plt.savefig(f"Result/Figure/close{stock}.png", dpi=300)
    else:
        pass

    # Show the plot
    plt.show()

# Visualize the closing price history with bull/bear power
def plot_bull_bear(stock, df, show=120, save=False):
    # Add technical indicators to the data
    add_indicator(df)

    # Filter the data
    df = df[- show:]

    # Define the widths
    width_candle = 1
    width_stick = 0.2

    # Separate the dataframe into green and red candlesticks
    up_df = df[df["Close"] > df["Open"]]
    down_df = df[df["Close"] <= df["Open"]]
    colour_up = "green"
    colour_down = "red"

    # Separate the dataframe into bull and bear power
    bull_df = df[df["Bull Bear"] >= 0]
    bear_df = df[df["Bull Bear"] < 0]
    colour_bull = "green"
    colour_bear = "red"

    # Create a figure with two subplots, one for the closing price and one for the volume
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [5, 1]}, sharex=True)

    # Plot the up prices on the top subplot
    ax1.bar(up_df.index, up_df["Close"] - up_df["Open"], width_candle, bottom=up_df["Open"], color=colour_up)
    ax1.bar(up_df.index, up_df["High"] - up_df["Close"], width_stick, bottom=up_df["Close"], color=colour_up)
    ax1.bar(up_df.index, up_df["Low"] - up_df["Open"], width_stick, bottom=up_df["Open"], color=colour_up)

    # Plot the down prices on the top subplot
    ax1.bar(down_df.index, down_df["Close"] - down_df["Open"], width_candle, bottom=down_df["Open"], color=colour_down)
    ax1.bar(down_df.index, down_df["High"] - down_df["Open"], width_stick, bottom=down_df["Open"], color=colour_down)
    ax1.bar(down_df.index, down_df["Low"] - down_df["Close"], width_stick, bottom=down_df["Close"], color=colour_down)

    # Plot the SMAs on the top subplot
    periods = [5, 20, 50, 200]
    for i in periods:
        ax1.plot(df[f"SMA {str(i)}"], label=f"SMA {str(i)}")

    # Set the y label of the top subplot
    ax1.set_ylabel("Price")

    # Plot the bull/bear power on the bottom subplot
    ax2.bar(bull_df.index, bull_df["Bull Bear"], label="Bull Power", color=colour_bull)
    ax2.bar(bear_df.index, bear_df["Bull Bear"], label="Bear Power", color=colour_bear)

    # Set the y label of the bottom subplot
    ax2.set_ylabel("Bull Bear Power")

    # Set the x limit of the top subplot
    buffer = relativedelta(days=1)
    ax1.set_xlim(df.index[0] - buffer, df.index[-1] + buffer)

    # Set the x label
    plt.xlabel("Date")

    # Set the title
    plt.suptitle(f"Closing price history for {stock}")

    # Combine the legends and place them at the top subplot
    handles, labels = ax1.get_legend_handles_labels()
    handles += ax2.get_legend_handles_labels()[0]
    labels += ax2.get_legend_handles_labels()[1]
    ax1.legend(handles, labels)

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Save the plot
    if save:
        plt.savefig(f"Result/Figure/bullbear{stock}.png", dpi=300)
    else:
        pass

    # Show the plot
    plt.show()

# Plot the MFI/RSI indicator
def plot_MFI_RSI(stock, df, show=252, save=False):
    # Add technical indicators to the data
    add_indicator(df)

    # Filter the data
    df = df[- show:]

    # Create a figure with three subplots, one for the closing price, one for the MFI/RSI indicator, and one for the MFI Z-Score
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [5, 1]}, sharex=True)

    # Plot the closing price on the firsst subplot
    ax1.plot(df["Close"], label="Close")

    # Set the y label of the first subplot
    ax1.set_ylabel("Price")

    # Set the x limit of the first subplot
    ax1.set_xlim(df.index[0], df.index[-1])

    # Plot the MFI/RSI indicator on the second subplot
    ax2.plot(df["MFI"], label="MFI", color="orange", alpha=0.7)
    ax2.plot(df["RSI"], label="RSI", color="green", alpha=0.7)
    ax2.axhline(y=20, linestyle="dotted", label="Oversold", color="red")
    ax2.axhline(y=80, linestyle="dotted", label="Overbought", color="red")

    # Set the y label of the second subplot
    ax2.set_ylabel(f"MFI/RSI")

    # Set the x label
    plt.xlabel("Date")
    
    # Set the title
    plt.suptitle(f"MFI/RSI for {stock}")

    # Combine the legends and place them at the top subplot
    handles, labels = ax1.get_legend_handles_labels()
    handles += ax2.get_legend_handles_labels()[0]
    labels += ax2.get_legend_handles_labels()[1]
    ax1.legend(handles, labels)

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Save the plot
    if save:
        plt.savefig(f"Result/Figure/MFIRSI{stock}.png", dpi=300)
    else:
        pass

    # Show the plot
    plt.show()

# Plot the follow-through days (FTDs) and distribution days (DDs)
def plot_FTD_DD(stock, df, show=252*2, save=False):
    # Add technical indicators to the data
    add_indicator(df)
    
    # Filter the data
    df = df[- show:]
    
    # Create a figure
    plt.figure(figsize=(10, 6))

    # Plot the closing price
    plt.plot(df["Close"])

    # Plot the FTDs
    plt.scatter(df.index[df["FTD"]], df["Close"][df["FTD"]], marker="x", color="green")

    # Plot the DDs
    plt.scatter(df.index[df["DD"]], df["Close"][df["DD"]], marker="x", color="red")
    
    # Plot if there are at least four follow-through days over the past month
    plt.scatter(df.index[df["Multiple FTDs"]], df["Close"][df["Multiple FTDs"]] - 10, marker="d", color="green")
    
    # Plot if there are at least four distribution days over the past month
    plt.scatter(df.index[df["Multiple DDs"]], df["Close"][df["Multiple DDs"]] + 10, marker="d", color="red")

    # Set the x limit
    plt.xlim(df.index[0], df.index[-1])

    # Set the labels
    plt.xlabel("Date")
    plt.ylabel("Price")

    # Set the title
    plt.title(f"Follow-through days and distribution days for {stock}")

    # Set the legend
    plt.legend(["Close", "follow-through day", "distribution day", "multiple follow-through days", "multiple distribution days"])

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Save the plot
    if save:
        plt.savefig(f"Result/Figure/FTDDD{stock}.png", dpi=300)
    else:
        pass

    # Show the plot
    plt.show()

# Plot the market breadth indicators
def plot_market_breadth(index_name, index_df, tickers, periods=[20, 50, 200], show=120, save=False):
    # Add technical indicators to the data
    add_indicator(index_df)

    # Filter the data
    index_df = index_df[- show:]

    # Define the widths
    width_candle = 1
    width_stick = 0.2

    # Separate the dataframe into green and red candlesticks
    up_df = index_df[index_df["Close"] > index_df["Open"]]
    down_df = index_df[index_df["Close"] <= index_df["Open"]]
    colour_up = "green"
    colour_down = "red"

    # Create a figure with three subplots, one for the closing price, one for the SMAs, and one for the AD line
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1, 1]}, sharex=True)

    # Plot the up prices on the first subplot
    ax1.bar(up_df.index, up_df["Close"] - up_df["Open"], width_candle, bottom=up_df["Open"], color=colour_up)
    ax1.bar(up_df.index, up_df["High"] - up_df["Close"], width_stick, bottom=up_df["Close"], color=colour_up)
    ax1.bar(up_df.index, up_df["Low"] - up_df["Open"], width_stick, bottom=up_df["Open"], color=colour_up)

    # Plot the down prices on the first subplot
    ax1.bar(down_df.index, down_df["Close"] - down_df["Open"], width_candle, bottom=down_df["Open"], color=colour_down)
    ax1.bar(down_df.index, down_df["High"] - down_df["Open"], width_stick, bottom=down_df["Open"], color=colour_down)
    ax1.bar(down_df.index, down_df["Low"] - down_df["Close"], width_stick, bottom=down_df["Close"], color=colour_down)
    
    # Set the label of the first subplot
    ax1.set_ylabel("Price")

    # Set the x limit of the first subplot
    ax1.set_xlim(index_df.index[0], index_df.index[-1])

    # Plot the % of tickers above the SMAs on the second subplot
    for i in periods:
        ax2.plot(index_df.index, index_df[f"Above SMA {str(i)}"] / len(tickers) * 100, label=f"% above SMA {str(i)}")

    # Set the y label of the second subplot
    ax2.set_ylabel(f"% above SMA")

    # Plot the AD line on the third subplot
    index_df["AD"] = index_df["AD"] - index_df["AD"].iloc[0]
    ax3.plot(index_df.index, index_df["AD"], color="red")

    # Set the y label of the third subplot
    ax3.set_ylabel("AD line")

    # Set the x label
    plt.xlabel("Date")

    # Set the title
    plt.suptitle(f"Market breadth of {index_name}")

    # Combine the legends and place them at the first subplot
    handles, labels = ax1.get_legend_handles_labels()
    handles += ax2.get_legend_handles_labels()[0]
    labels += ax2.get_legend_handles_labels()[1]
    ax1.legend(handles, labels)

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Save the plot
    if save:
        plt.savefig(f"Result/Figure/marketbreadth{index_name}.png", dpi=300)
    else:
        pass

    # Show the plot
    plt.show()

# Plot to compare the closing price history of stocks
def plot_stocks(stocks, current_date, column="Close", show=120, save=False):
    # Merge dataframes of stocks
    df_merged = merge_stocks(stocks, current_date)

    # Restrict the dataframe
    df_merged = df_merged[- show:]

    # Create a figure
    plt.figure(figsize=(10, 6))

    # Plot the closing price history of the stocks
    for stock in stocks:
        close_first = df_merged[f"{column} ({stock})"].iloc[0]
        plt.plot(100 / close_first * df_merged[f"{column} ({stock})"], label=f"{stock} (scaled)")

    # Set the x limit
    plt.xlim(df_merged.index[0], df_merged.index[-1])

    # Set the labels
    plt.xlabel("Date")
    plt.ylabel("Price")

    # Set the legend
    plt.legend()

    # Set the title
    plt.title(f"Closing price history for {', '.join(stocks)}")
    
    # Adjust the spacing
    plt.tight_layout()

    # Save the plot
    if save:
        plt.savefig("Result/Figure/closestocks.png", dpi=300)
    else:
        pass

    # Show the plot
    plt.show()

# Plot the JdK RS-Ratio and Momentum of a sector
def plot_JdK(sector, sector_dict, index_df, show=120, save=False):
    # Filter the data
    index_df = index_df[- show:]

    # Extract the columns
    columns = [f"{sector} JdK RS-Ratio", f"{sector} JdK RS-Momentum"]

    # Create a figure with two subplots, one for the JdK RS-Ratio and one for JdK RS-Momentum
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [1, 1]}, sharex=True)

    # Plot the JdK RS-Ratio on the top subplot
    ax1.plot(index_df[columns[0]], label=columns[0])
    ax1.axhline(y=100, linestyle="dotted", color="black")

    # Set the y label of the top subplot
    ax1.set_ylabel("JdK RS-Ratio")

    # Set the x limit of the top subplot
    ax1.set_xlim(index_df.index[- show], index_df.index[-1])

    # Set the legend of the top subplot
    ax1.legend()

    # Plot the JdK RS-Momentum on the bottom subplot
    ax2.plot(index_df[columns[1]], label=columns[1])

    # Add a horizontal dotted line at 100 to the bottom subplot
    ax2.axhline(y=100, linestyle="dotted", color="black")

    # Set the y label of the bottom subplot
    ax2.set_ylabel("JdK RS-Momentum")

    # Set the legend of the bottom subplot
    ax2.legend()

    # Set the x label
    plt.xlabel("Date")

    # Set the title
    plt.suptitle(f"JdK RS-ratio and momentum for {sector_dict[sector]}")

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Save the plot
    if save:
        plt.savefig(f"Result/Figure/JdKRS{sector}.png", dpi=300)
    else:
        pass

    # Show the plot
    plt.show()

# Plot the relative rotation graph
def plot_rrg(sectors, sector_dict, index_df, type, points=8, interval=5, save=False):
    # Define the colors
    colors = plt.cm.tab10(range(10)).tolist() + ["peru", "navy", "mediumspringgreen", "olivexs"]

    # Create a figure and axes
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Initialize two empty lists to store the data points
    xs = []
    ys = []

    # Plot the JdK RS-Ratio and Momentum for each sector
    for i, sector in enumerate(sectors):
        if sector == "^GSPC":
            continue
        color = colors[i]
        label = sector_dict[sector]

        # Scatter the points
        for point in range(points):
            x = index_df[f"{sector} JdK RS-Ratio"].iloc[- 1 - point * interval]
            y = index_df[f"{sector} JdK RS-Momentum"].iloc[- 1 - point * interval]
            xs.append(x)
            ys.append(y)
            label = sector_dict[sector]
            if point == 0:
                ax1.scatter(x, y, color=color, s=50, marker=">", label=label)
            elif point == points - 1:
                ax1.scatter(x, y, color=color, s=50, marker="o")
            else:
                ax1.scatter(x, y, color=color, s=10, marker="o")

            # Connect the point with dashed lines
            if point > 0:
                ax1.plot([x_prev, x], [y_prev, y], color=color, linestyle="--")
            x_prev, y_prev = x, y

    # Set the labels
    ax1.set_xlabel("JdK RS-Ratio")
    ax1.set_ylabel("JdK RS-Momentum")

    # Set the title
    if type == "sector":
        ax1.set_title("Relative rotation graph of sectors")
    elif type == "index":
        ax1.set_title("Relative rotation graph of indices")

    # Add horizontal and vertical lines to (100, 100) origin
    ax1.axhline(y=100, linestyle="--", color="black")
    ax1.axvline(x=100, linestyle="--", color="black")

    # Set the limits
    buffer = 0.25
    x_min, x_max = min(xs) - buffer, max(xs) + buffer
    y_min, y_max = min(ys) - buffer, max(ys) + buffer
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)

    # Colour each quadrant
    ax1.fill_between([100, x_max], [100, 100], [y_max, y_max], color="green", alpha=0.1)
    ax1.fill_between([x_min, 100], [100, 100], [y_max, y_max], color="blue", alpha=0.1)
    ax1.fill_between([100, x_max], [y_min, y_min], [100, 100], color="gold", alpha=0.1)
    ax1.fill_between([x_min, 100], [y_min, y_min], [100, 100], color="red", alpha=0.1)

    # Add text labels in each corner
    ax1.text(x_max, y_max, "Leading", color="green", ha="right", va="top", weight="bold")
    ax1.text(x_min, y_max, "Improving", color="blue", ha="left", va="top", weight="bold")
    ax1.text(x_max, y_min, "Weakening", color="gold", ha="right", va="bottom", weight="bold")
    ax1.text(x_min, y_min, "Lagging", color="red", ha="left", va="bottom", weight="bold")

    # Set the legend outside the plot
    ax1.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0, fontsize=8)

    # Adjust the spacing between subplots
    plt.tight_layout()
    
    # Save the plot
    if save:
        if type == "sector":
            plt.savefig(f"Result/Figure/sectorrrg.png", dpi=300)
        elif type == "index":
            plt.savefig(f"Result/Figure/indexrrg.png", dpi=300)
    else:
        pass

    # Show the plot
    plt.show()

# Plot the sectors of the selected stocks
def plot_sector_selected(end_date, index_name, index_dict, period=252, RS=90, NASDAQ_all=True, save=False):
    # Get the infix
    infix = get_infix(index_name, index_dict, NASDAQ_all)
    
    # Format the end date
    end_date_fmt = dt.datetime.strptime(end_date, "%Y-%m-%d").strftime("%d-%m-%y")

    # Define the folder path
    folder_path = os.path.join("Result", f"{end_date_fmt}")

    # Define the filename
    filename = os.path.join(folder_path, f"{infix}stock_{end_date_fmt}period{period}RS{RS}.xlsx")
    
    # Read the data of the screened stocks
    df = pd.read_excel(filename)

    # Count the occurrences of each sector
    sector_counts = df["Sector"].value_counts()

    # Customize the colours
    colors = plt.cm.tab10(range(10)).tolist() + ["peachpuff"]

    # Create a pie chart with count numbers
    plt.figure(figsize=(8, 6))
    plt.pie(sector_counts, labels=sector_counts.index, autopct=lambda x: f'{int(round(x*sum(sector_counts)/100))}', colors=colors)

    # Set the title
    plt.title("Sector distribution of selected stocks")

    # Set the axes to be equal
    plt.axis("equal")

    # Adjust the spacing
    plt.tight_layout()

    # Save the plot
    if save:
        plt.savefig(f"Result/Figure/{infix}sectorselected.png", dpi=300, bbox_inches="tight")
    else:
        pass

    # Show the plot
    plt.show()

# Plot the correlation matrix of technical indicators
def plot_corr_ta(stock, df, column_list=["Open", "High", "Low", "Close", "Volume", "MACD", "RSI", "RMI", "CCI", "ADX", "MFI", "OBOS"]):
    # Extract the data
    data = df.copy().dropna()[column_list].values

    # Calculate the correlation matrix
    correlation_matrix = np.corrcoef(data, rowvar=False)

    # Create a heatmap to visualize the correlation matrix
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", xticklabels=column_list, yticklabels=column_list)
    plt.title(f"Correlation matrix of techinical indicators of {stock}")
    plt.show()

# Plot the correlation matrix of stocks
def plot_corr_stocks(stocks, end_date, years):
    # Get the price data of the stocks
    df_merged = merge_stocks(stocks, end_date)

    # Restrict the dataframe
    show = int(years * 252)
    df_merged = df_merged[- show:]
    dfs_close = [df_merged["Close"].values]
    for i in range(1, len(stocks)):
        dfs_close.append(df_merged[f"Close ({stocks[i]})"].values)

    # Create the data with the aligned values
    data = np.array(dfs_close)

    # Calculate the correlation matrix
    correlation_matrix = np.corrcoef(data)
    
    # Create a heatmap
    tick_labels = stocks
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", xticklabels=tick_labels, yticklabels=tick_labels)

    # Set the title
    if years == 1:
        plt.title(f"Correlation matrix in the past {years} year")
    else:
        plt.title(f"Correlation matrix in the past {years} years")

    # Show the plot
    plt.show()

# Plot the autocorrelation of a stock
def plot_autocorr(stock, end_date, years):
    # Get the start date
    start_date = (dt.datetime.strptime(end_date, "%Y-%m-%d") - relativedelta(years=years)).strftime("%Y-%m-%d")

    # Get the price data of the stock
    df = get_df(stock, end_date)

    # Filter the data
    df = df[start_date : end_date]

    # Drop rows with nan values and get the closing prices
    data = df.dropna()["Close"].values

    # Create a figure
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    # Calculate the autocorrelation
    acfs = acf(data, nlags=252*5)

    # Plot the autocorrelation
    ax.plot(np.arange(len(acfs)), acfs)
    
    # Find the local maxima
    maxima_indices = argrelextrema(acfs, np.greater)[0]
    maxima_values = acfs[maxima_indices]
    for index, value in zip(maxima_indices, maxima_values):
        print(f"Index: {index}, Value: {value}")
        
    # Mark the local maxima on the plot
    ax.plot(maxima_indices, maxima_values, "rx", label="Local maxima")

    # Set the x limit
    plt.xlim(0, 252 * years)

    # Set the title
    if years == 1:
        plt.title(f"Autocorrelation function for {stock} in the past {years} year")
    else:
        plt.title(f"Autocorrelation function for {stock} in the past {years} years")

    # Adjust the spacing
    plt.tight_layout()

    # Set the legend
    plt.legend()

    # Show the plot
    plt.show()