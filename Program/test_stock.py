# Imports
import datetime as dt
from helper_functions import generate_end_dates, stock_market, merge_stocks
import pandas as pd
from plot import *
from stock_screener import EM_rating, stoploss_target
from technicals import *

# Plot colours
colors = plt.cm.tab10.colors 

# Start of the program
start = dt.datetime.now()

# Initial setup
current_date = start.strftime("%Y-%m-%d")

# Choose the stocks
stocks = ["CORZ", "HBM", "PLTR", "PGR", "VRT", "1810.HK", "3690.HK", "3998.HK"]

# # Iterate over stocks
# for stock in stocks:
#     df = get_df(stock, current_date)
#     plot_close(stock, df, save=True)
#     plot_MFI_RSI(stock, df, save=True)
#     plot_stocks(["^GSPC", stock], current_date, save=True)

# # Get the stop loss and target price of a stock
# stock = "HBM"
# df = get_df(stock, current_date)
# current_close = df["Close"].iloc[-1]
# stoploss, stoploss_pct, target, target_pct = stoploss_target(stock, 9.38, current_date)
# print(f"Plan for {stock}.")
# print(f"Current close: {round(current_close, 2)}.")
# print(f"Stoploss: {stoploss}, {stoploss_pct} (%).")
# print(f"Target price: {target}, {target_pct} (%).")

compare_metal = False
if compare_metal:
    show = 252 * 3
    stocks = ["GC=F", "SI=F", "HG=F"]
    metal_df = merge_stocks(stocks, current_date)
    metal_df["Gold/Silver Ratio"] = metal_df["Close (GC=F)"] / metal_df["Close (SI=F)"]
    metal_df["Gold/Copper Ratio"] = metal_df["Close (GC=F)"] / metal_df["Close (HG=F)"]
    metal_df = calculate_ZScore(metal_df, ["Gold/Silver Ratio", "Gold/Copper Ratio"], 252)

    # Restrict the dataframe
    metal_df = metal_df[- show:]

    # Create a figure with three subplots, one for the metal prices, one for the ratios, one for the ratios z-score
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1, 1]}, sharex=True)

    # Plot the metal prices on the first subplot
    close_goldfirst = metal_df["Close (GC=F)"].iloc[0]
    close_silverfirst = metal_df["Close (SI=F)"].iloc[0]
    close_copperfirst = metal_df["Close (HG=F)"].iloc[0]
    ax1.plot(100 / close_goldfirst * metal_df["Close (GC=F)"], label="Gold (scaled)", color="gold")
    ax1.plot(100 / close_silverfirst * metal_df["Close (SI=F)"], label="Silver (scaled)", color="silver")
    ax1.plot(100 / close_copperfirst * metal_df["Close (HG=F)"], label="Copper (scaled)", color="peru")

    # Set the label of the first subplot
    ax1.set_ylabel("Price")

    # Set the x limit of the first subplot
    ax1.set_xlim(metal_df.index[0], metal_df.index[-1])

    # Plot the ratios on the second subplot
    goldsilver_ratio_first = metal_df["Gold/Silver Ratio"].iloc[0]
    goldcopper_ratio_first = metal_df["Gold/Copper Ratio"].iloc[0]
    ax2.plot(100 / goldsilver_ratio_first * metal_df["Gold/Silver Ratio"], color="silver")
    ax2.plot(100 / goldcopper_ratio_first * metal_df["Gold/Copper Ratio"], color="peru")

    # Set the y label of the second subplot
    ax2.set_ylabel("Ratio wrt Gold")

    # Plot the ratios z-score on the third subplot
    ax3.plot(metal_df["Gold/Silver Ratio Z-Score"], color="silver")
    ax3.plot(metal_df["Gold/Copper Ratio Z-Score"], color="peru")
    ax3.axhline(y=2, linestyle="dotted", label="Undervalued", color="green")
    ax3.axhline(y=-2, linestyle="dotted", label="Overvalued", color="red")

    # Set the y label of the second subplot
    ax3.set_ylabel("Ratio z-score")

    # Set the x label
    plt.xlabel("Date")

    # Set the title
    plt.suptitle(f"Metal prices comparison")

    # Combine the legends and place them at the top subplot
    handles, labels = ax1.get_legend_handles_labels()
    handles += ax3.get_legend_handles_labels()[0]
    labels += ax3.get_legend_handles_labels()[1]
    ax1.legend(handles, labels)

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Save the plot
    plt.savefig("Result/Figure/metalcompare.png", dpi=300)    

    # Show the plot
    plt.show()

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
    stocks = ["^HSI", "9618.HK"]

    # Get the price data of 京東 and HSI
    df_merged = merge_stocks(stocks, current_date)
    df_merged["HSI Percent Change"] = df_merged["Close (^HSI)"].pct_change()
    df_merged["JD Percent Change"] = df_merged["Close (9618.HK)"].pct_change()

    # Restrict the dataframe
    show = 252
    df_merged = df_merged[- show:]

    # Calculate the correlation factor
    hsi_pct = df_merged["HSI Percent Change"].to_numpy()
    jd_pct = df_merged["JD Percent Change"].to_numpy()
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
    df_merged["Leveraged Close"] = (df_merged["Close (^HSI)"].iloc[0] * (1 + min_factor * df_merged["HSI Percent Change"]).cumprod())

    # Get the first closing price of the first stock
    close_first0 = df_merged["Leveraged Close"].iloc[0]

    # Plot the closing price history of the first stock
    plt.plot(100 / close_first0 * df_merged["Leveraged Close"], label=f"{min_factor:.2f}x HSI (scaled)")

    # Plot the closing price history
    close_first = df_merged["Close (9618.HK)"].iloc[0]
    plt.plot(100 / close_first * df_merged["Close (9618.HK)"], label="9618.HK (scaled)")

    # Set the x limit
    plt.xlim(df_merged.index[0], df_merged.index[-1])

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