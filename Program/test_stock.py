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

# Choose the stocks
stocks = ["ACIW", "EGO", "PGR"]

# Iterate over stocks
for stock in stocks:
    df = get_df(stock, current_date)
    plot_close(stock, df, save=True)
    plot_MFI_RSI(stock, df, save=True)
    plot_stocks(["^GSPC", "^IXIC", stock], current_date, save=True)

# Get the stop loss and target price of a stock
stock = "EGO"
stoploss, stoploss_pct, target, target_pct = stoploss_target(stock, 16.35, "2024-09-11")
print(f"Stoploss: {stoploss}, {stoploss_pct} (%).")
print(f"Target price: {target}, {target_pct} (%).")

# Print the end time and total runtime
end = dt.datetime.now()
print(end, "\n")
print("The program used", end - start)