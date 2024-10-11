# Imports
import datetime as dt
from helper_functions import generate_end_dates, stock_market
import pandas as pd
from plot import *
from stock_screener import EM_rating, stoploss_target
from technicals import *

# Start of the program
start = dt.datetime.now()

# Initial setup
current_date = start.strftime("%Y-%m-%d")

# # Choose the stocks
# stocks = ["ACIW", "1810.HK", "3690.HK", "3998.HK"]

# # Iterate over stocks
# for stock in stocks:
#     df = get_df(stock, current_date, redownload=True)
#     plot_close(stock, df, save=True)
#     plot_MFI_RSI(stock, df, save=True)
#     plot_stocks(["^GSPC", "^GSPC", stock], current_date, save=True)

# Get the stop loss and target price of a stock
stock = "HBM"
df = get_df(stock, current_date)
current_close = df["Close"].iloc[-1]
stoploss, stoploss_pct, target, target_pct = stoploss_target(stock, 9.38, current_date)
print(f"Plan for {stock}.")
print(f"Current close: {round(current_close, 2)}.")
print(f"Stoploss: {stoploss}, {stoploss_pct} (%).")
print(f"Target price: {target}, {target_pct} (%).")

hsi_mfirsi = False
if hsi_mfirsi:
    # Choose the stock
    stock = "^HSI"

    # Get the price data of the stock
    df = get_df(stock, current_date)

    # Add indicators
    df = add_indicator(df)
    df = calculate_ZScore(df, ["MFI", "RSI"], period=252*15)

    # Save the data of the index to a .csv file
    filename = f"Price data/{stock}_{current_date}.csv"
    df.to_csv(filename)

    periods = [5, 10, 15, 20, 30, 60]
    for period in periods:
        df[f"Close {period} Later"] = df["Close"].shift(- period)
        df[f"{period} Days Return (%)"] = ((df[f"Close {period} Later"] / df["Close"]) - 1) * 100

    # Filter for MFI/RSI Z-Score >= 2.5
    df_MFIRSI_filter = df[(df["MFI Z-Score"] >= 2.5)]
    print(df_MFIRSI_filter)

    # Plot histogram
    for period in periods:
        # Create a figure
        plt.figure(figsize=(10, 6))

        # Plot the histogram
        plt.hist(df_MFIRSI_filter[f"{period} Days Return (%)"].dropna(), bins=30, label=f'{period} Days Return (%)')

        # Calculate the mean
        mean = df_MFIRSI_filter.loc[:, f"{period} Days Return (%)"].mean()

        # Draw a vertical line at the mean
        plt.axvline(mean, color="red", linestyle="dashed", linewidth=1.5, label=f"Mean: {mean:.2f}%")

        # Set the y-axis ticks to integers
        y_ticks = np.arange(0, plt.ylim()[1] + 1, 1)
        plt.yticks(y_ticks)

        # Set the labels
        plt.xlabel("Return (%)")
        plt.ylabel("Count")

        # Set the title
        plt.title(rf"{period} days return when MFI Z-Score$\geq 2.5$ (%)")

        # Set the legend
        plt.legend()

        # Adjust the spacing
        plt.tight_layout()

        # Save the plot
        plt.savefig(f"Result/Figure/{period}returnMFIZgeq2.5.png", dpi=300)

        # Show the plot
        plt.show()

compare_jdhsi = False
if compare_jdhsi:
    # Test the correlation between 京東 and HSI
    jd_df = get_df("9618.HK", current_date, redownload=True)
    hsi_df = get_df("^HSI", current_date, redownload=True)

    # Calculate percentage change for both
    show = 120
    hsi_df["HSI Percent Change"] = hsi_df["Close"].pct_change()
    jd_df["JD Percent Change"] = jd_df["Close"].pct_change()

    # Merge dataframes based on index
    merged_df = pd.merge(hsi_df, jd_df["JD Percent Change"], left_index=True, right_index=True, how="inner")
    merged_df = merged_df.dropna()

    # Select the last 'show' periods
    merged_df_res = merged_df[- show:]

    # Calculate the correlation factor
    hsi_pct = merged_df_res["HSI Percent Change"].to_numpy()
    jd_pct = merged_df_res["JD Percent Change"].to_numpy()
    factor_corr = np.corrcoef(hsi_pct, jd_pct)[0, 1]
    print(f"The correlation factor between HSI and 9618.HK is {factor_corr:.2f}.")

    # Iterate over all factors
    factors = np.arange(0, 3.01, 0.01)
    diff_sums = np.zeros(len(factors))
    for i, factor in enumerate(factors):
        diff_sums[i] = np.sum((jd_pct - factor * hsi_pct)**2)

    # Find the factor of the minimum difference sum
    min_index = np.argmin(diff_sums)
    min_factor = factors[min_index]
    min_diff_sums = diff_sums[min_index]
    print(f"The HSI best approximates 9618.HK at a factor of {min_factor}, with a minimum difference of {min_diff_sums}.")

    # Create a figure
    plt.figure(figsize=(10, 6))

    # Create a leveraged hsi_df
    hsi_df["Leveraged Close"] = (hsi_df["Close"].iloc[0] * (1 + min_factor * hsi_df["HSI Percent Change"]).cumprod())

    # Select the last 'show' periods for plotting
    hsi_df = hsi_df[- show:].dropna()
    jd_df = jd_df[- show:].dropna()

    # Get the first closing price of the first stock
    close_first0 = hsi_df["Leveraged Close"].iloc[0]

    # Plot the closing price history of the first stock
    plt.plot(100 / close_first0 * hsi_df["Leveraged Close"], label=f"{min_factor:.2f}x HSI (scaled)")

    # Plot the closing price history
    close_first = jd_df["Close"].iloc[0]
    plt.plot(100 / close_first * jd_df["Close"], label="9618.HK (scaled)")

    # Set the x limit
    plt.xlim(hsi_df.index[0], hsi_df.index[-1])

    # Set the labels
    plt.xlabel("Date")
    plt.ylabel("Price")

    # Set the legend
    plt.legend()

    # Set the title
    plt.title("Closing price history for stocks")

    # Adjust the spacing
    plt.tight_layout()

    # Save the plot
    plt.savefig(f"Result/Figure/HSIJDperiod{show}.png", dpi=300)

    # Show the plot
    plt.show()

# Print the end time and total runtime
end = dt.datetime.now()
print(end, "\n")
print("The program used", end - start)