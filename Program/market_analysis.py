# Import
import datetime as dt
from helper_functions import get_df, stock_market
from plot import *
from technicals import *

# Main function
def main():
    # Start of the program
    start = dt.datetime.now()

    # Initial setup
    current_date = start.strftime("%Y-%m-%d")
    current_date = "2024-08-24"

    # Variables
    NASDAQ_all = True

    # Index
    index_name = "^GSPC"
    index_names = ["^GSPC", "^IXIC", "^DJI", "IWM", "FFTY", "QQQE"]
    index_dict = {"^GSPC": "S&P 500", "^IXIC": "NASDAQ Composite", "^DJI": "Dow Jones Industrial Average", 
                  "IWM": "iShares Russell 2000 ETF", "FFTY": "Innovator IBD 50 ETF", "QQQE": "NASDAQ-100 Equal Weighted ETF"}
    
    # Sectors
    sectors = ["XLC", "XLY", "XLP", "XLE", "XLF", "XLV", 
            "XLI", "XLB", "XLRE", "XLK", "XLU"]
    sector_dict = {"XLC": "Communication Services", "XLY": "Consumer Discretionary", "XLP": "Consumer Staples", 
                "XLE": "Energy", "XLF": "Financials", "XLV": "Health Care", 
                "XLI": "Industrials", "XLB": "Materials", "XLRE": "Real Estate", 
                "XLK": "Technology", "XLU": "Utilities"}
    
    # Get the price data of the index
    index_df = get_df(index_name, current_date)

    # Iterate over all indices and sectors
    for ticker in index_names + sectors:
        # Get the price data of the tickers
        df = get_df(ticker, current_date)

        # Visualize the closing price history of the ticker
        plot_close(ticker, df, MVP_VCP=False, save=True)

    # Calculate the JdK RS-Ratio and Momentum
    index_df = get_JdK(sectors, index_df, current_date)

    # Print the leading, weakening, improving and lagging sectors
    sectors_leading = []
    sectors_weakening = []
    sectors_improving = []
    sectors_lagging = []

    # Iterate over all sectors
    for sector in sectors:
        if index_df[f"{sector} JdK RS-Ratio"].iloc[-1] > 100:
            if index_df[f"{sector} JdK RS-Momentum"].iloc[-1] > 100:
                sectors_leading.append(sector)
            else:
                sectors_weakening.append(sector)
        else:
            if index_df[f"{sector} JdK RS-Momentum"].iloc[-1] > 100:
                sectors_improving.append(sector)
            else:
                sectors_lagging.append(sector)

    # Print the classified sectors
    print(f"Leading sectors: {', '.join(sectors_leading)}")
    print(f"Weakening sectors: {', '.join(sectors_weakening)}")
    print(f"Improving sectors: {', '.join(sectors_improving)}")
    print(f"Lagging sectors: {', '.join(sectors_lagging)}")

    # Iterate over all sectors
    for sector in sectors:
        # Plot the JdK RS-Ratio and Momentum of the sector
        plot_JdK(sector, sector_dict, index_df, save=True)

    # Plot the relative rotation graph
    plot_rrg(sectors, sector_dict, index_df, save=True)

    # Plot the sectors of the selected stocks
    plot_sector_selected(current_date, "^GSPC", index_dict, NASDAQ_all=NASDAQ_all, save=True)

    # Get the list of tickers of stock market
    tickers = stock_market(current_date, current_date, index_name, False)

    # Calculate the market breadth indicators
    index_df = market_breadth(current_date, index_df, tickers)

    # Save the data of the index to a .csv file
    filename = f"Price data/{index_name}_{current_date}.csv"
    index_df.to_csv(filename)

    # Visualize the closing price history and other technical indicators
    plot_market_breadth(index_name, index_df, tickers, save=True)
    plot_close(index_name, index_df, MVP_VCP=False)
    plot_bull_bear(index_name, index_df, save=True)
    plot_MFI_RSI(index_name, index_df, save=True)
    plot_FTD_DD(index_name, index_df, save=True)
 
    # Get the price data of CBOE Volatility Index (VIX)
    vix_df = get_df("^VIX", current_date)

    # Get the current closing price of VIX
    vix_current_high = round(vix_df["High"].iloc[-1], 2)

    # Define the exit indicator based on VIX value
    if vix_current_high < 26:
        vix_colour = "Green"
    elif 26 < vix_current_high < 30:
        vix_colour = "Yellow"
    elif vix_current_high > 30:
        vix_colour = "Red"

    # Plot the closing price history of VIX
    plot_close("^VIX", vix_df, save=True)

    # Print the exit indicator
    print(f"Current VIX: {vix_current_high} ({vix_colour})")

    # Print the end time and total runtime
    end = dt.datetime.now()
    print(end, "\n")
    print("The program used", end - start)

# Run the main function
if __name__ == "__main__":
    main()