# Imports
import datetime as dt
from helper_functions import generate_end_dates, stock_market
import pandas as pd
from plot import *
# Set display options to show all rows and columns
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
from stock_screener import EM_rating, stoploss_target
from technicals import *

# Start of the program
start = dt.datetime.now()

# Initial setup
current_date = start.strftime("%Y-%m-%d")
current_date = "2024-10-03"

# # Choose the stocks
# stocks = ["0700.HK", "1810.HK", "1339.HK", "2618.HK", "3690.HK", "6881.HK"]

# # Iterate over stocks
# for stock in stocks:
#     df = get_df(stock, current_date)
#     plot_close(stock, df, show=200, save=True)
#     plot_MFI_RSI(stock, df, show=375, save=True)
#     plot_stocks(["^GSPC", "^GSPC", stock], current_date, save=True)

# # Get the stop loss and target price of a stock
# for stock in stocks:
#     df = get_df(stock, current_date)
#     current_close = df["Close"].iloc[-1]
#     stoploss, stoploss_pct, target, target_pct = stoploss_target(stock, current_close, current_date)
#     print(f"Current close: {round(current_close, 2)}.")
#     print(f"Stoploss: {stoploss}, {stoploss_pct} (%).")
#     print(f"Target price: {target}, {target_pct} (%).")

# Choose the stock
stock = "^HSI"

# Get the price data of the stock
df = get_df(stock, current_date)

# Add indicators
df = add_indicator(df)
df = calculate_ZScore(df, ["MFI", "RSI"], period=252*10)

# Save the data of the index to a .csv file
filename = f"Price data/{stock}_{current_date}.csv"
df.to_csv(filename)

periods = [5, 10, 20, 60]
for period in periods:
    df[f"Close {period} Later"] = df["Close"].shift(- period)
    df[f"{period} Days Return (%)"] = ((df[f"Close {period} Later"] / df["Close"]) - 1) * 100

# Filter for MFI/RSI Z-Score >= 2.5
df_MFIRSI_filter = df[(df["MFI Z-Score"] >= 2.5)]

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

# Print the end time and total runtime
end = dt.datetime.now()
print(end, "\n")
print("The program used", end - start)