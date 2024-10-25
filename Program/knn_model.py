# Imports
from backtest import calculate_stats, stocks_equity_curve
import datetime as dt
from dateutil.relativedelta import relativedelta
from helper_functions import get_current_date, generate_end_dates, get_df, get_infix
import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from technicals import *
from tqdm import tqdm

# Define the Lorentzian metric
def lorentzian_metric(arr1, arr2):
    # Compute the time difference
    dx0 = arr1[0] - arr2[0]

    # Compute the spatial difference vector
    dxi = arr1[1:] - arr2[1:]

    # Compute the Lorentzian distance
    distance =  - dx0 ** 2 + np.sum(dxi ** 2) + 10

    return distance

# Create the train and test data
def preprocess_knn(df, start_date, end_date, lookback, features, scaler=StandardScaler()):
    # Calculate the SMAs
    for i in [20, 50]:
        df[f"SMA {str(i)}"] = SMA(df, i)

        # Calculate the ratio of the closing price to the SMA
        df[f"SMA {str(i)} Ratio"] = df["Close"] / df[f"SMA {str(i)}"]

    # Add technical indicators to the data
    add_indicator(df)
    
    # Create a new column "Change" indicating if the closing price increases or decreases the next day
    df["Change"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    # Select the features
    X = df[features]
    X.head()

    # Target variable
    Y = df["Change"]

    # Define the start date for training, considering the lookback period
    train_start_date = (dt.datetime.strptime(start_date, "%Y-%m-%d") - relativedelta(years=lookback)).strftime("%Y-%m-%d")

    # Modify the training start date
    train_start_date = df.index[df.index >= train_start_date].min()

    # Define the end date for training
    train_end_date = start_date

    # Get the indices of dates
    train_start_index = len(df[ : train_start_date])
    train_end_index = len(df[ : train_end_date])
    end_index = len(df[ : end_date])

    # Train data
    X_train = X[train_start_index : train_end_index]

    # Scale the train data
    X_train = scaler.fit_transform(X_train)
    Y_train = Y[train_start_index : train_end_index]
    
    # Test data
    X_test = X[train_end_index : end_index + 1]

    # Scale the test data using the same scaler
    X_test = scaler.transform(X_test)
    Y_test = Y[train_end_index : end_index + 1]

    # Create a dataframe for testing
    df_test = df[train_end_date : end_date]

    return X_train, Y_train, X_test, Y_test, df_test

# Calculate the test accuracy of the KNN model
def knn_accuracy(X_train, Y_train, X_test, Y_test, k, lorentzian=False):
    if lorentzian:
        # Initiate the Lorentzian KNN model
        lknn = KNeighborsClassifier(n_neighbors=k, metric=lorentzian_metric)

        # Fit the model
        lknn.fit(X_train, Y_train)

        # Accuracy score
        X_train_lknn = lknn.predict(X_train)
        X_test_lknn = lknn.predict(X_test)
        accuracy_train_lknn = accuracy_score(Y_train, X_train_lknn)
        accuracy_test_lknn = accuracy_score(Y_test, X_test_lknn)

        # Confusion matrix
        cm_train_lknn = confusion_matrix(Y_train, X_train_lknn)
        cm_test_lknn = confusion_matrix(Y_test, X_test_lknn)
        
        return accuracy_train_lknn, accuracy_test_lknn, cm_train_lknn, cm_test_lknn, X_train_lknn, X_test_lknn
    
    else:
        # Initiate the KNN model
        knn = KNeighborsClassifier(n_neighbors=k)

        # Fit the model
        knn.fit(X_train, Y_train)

        # Accuracy score
        X_train_knn = knn.predict(X_train)
        X_test_knn = knn.predict(X_test)
        accuracy_train_knn = accuracy_score(Y_train, X_train_knn)
        accuracy_test_knn = accuracy_score(Y_test, X_test_knn)

        # Confusion matrix
        cm_train_knn = confusion_matrix(Y_train, X_train_knn)
        cm_test_knn = confusion_matrix(Y_test, X_test_knn)
        
        return accuracy_train_knn, accuracy_test_knn, cm_train_knn, cm_test_knn, X_train_knn, X_test_knn

# Define the main function
def main():
    # Start of the program
    start = dt.datetime.now()

    # Initial setup
    current_date = get_current_date(start)

    # Create the end dates
    end_dates = generate_end_dates(5, current_date)[:-1]
    end_dates.append("2024-07-10")

    # Variables
    NASDAQ_all = True
    factors = [0.05, 0.95, 0.0]

    # Index
    index_name = "^GSPC"
    index_dict = {"^GSPC": "S&P 500", "QQQ": "QQQ"}

    # Get the infix
    infix = get_infix(index_name, index_dict, NASDAQ_all)

    # Create a group of factors
    factors_group = []
    for i, j, k in itertools.product(range(20 + 1), repeat=3):
        if i + j + k == 20:
            factors_group.append([i / 20, j / 20, k / 20])

    # Parameters of the KNN model
    top = 5
    ks = range(1, 50 + 1)
    k = 3
    lookbacks = [1, 2]
    lookback = 1
    features = ["Close", "SMA 50 Ratio", "RSI", "RMI", "MFI", "ADX"]
    knn_params = {"k": k, "lookback": lookback, "features": features}

    # Get the equity curve of the KNN model
    index_df, cm_test_knn_index, cm_test_lknn_index = stocks_equity_curve(end_dates, current_date, index_name, index_dict, NASDAQ_all, factors, top, knn_params=knn_params)
        
    # Plot the equity curve of the KNN model
    # Create a figure
    plt.figure(figsize=(10, 6))

    # Plot the cumulative return of the index
    plt.plot(index_df["Cumulative Return"], label=index_dict[index_name])

    # Plot the cumulative return of the stocks
    plt.plot(index_df["Cumulative Stock Return"], label=f"Stocks {factors}")

    # Plot the cumulative return of the KNN model
    plt.plot(index_df["Cumulative KNN Stock Return"], label=f"KNN stocks {factors}")

    # Plot the cumulative return of the Lorentzian KNN model
    plt.plot(index_df["Cumulative LKNN Stock Return"], label=f"Lorentzian KNN stocks {factors}")

    # Set the labels
    plt.xlabel("Date")
    plt.ylabel("Cumulative return")

    # Set the x limit
    plt.xlim(index_df.index[0], index_df.index[-1])

    # Set the title
    plt.title("Equity curve")

    # Set the legend
    plt.legend(loc="upper left")

    # Save the plot
    plt.savefig(f"Result/Figure/{infix}equitycurvelknn{factors}k{k}lb{lookback}top{top}.png", dpi=300)

    # Show the plot
    plt.show()

    # Calculate the statistics of the Lorentzian KNN model
    returns, stats = calculate_stats(index_df, len(index_df) / 252, "LKNN Stock")
    total_return_stock = stats[0]
    return_peak = stats[1]
    CAGR_stock = stats[2]
    sharpe_ratio_stock = stats[4]
    sortino_ratio_stock = stats[5]

    # Print the results
    print(f"Total return: {total_return_stock:.2f}.")
    print(f"Peak of return: {return_peak:.2f}.")
    print(f"CAGR: {(CAGR_stock * 100):.2f}%.")
    print(f"Sharpe ratio: {sharpe_ratio_stock:.2f}.")
    print(f"Sortino ratio: {sortino_ratio_stock:.2f}.")

    # Get the price data of the index
    index_df = get_df(index_name, current_date)
    index_df = add_indicator(index_df)

    # Iterate over all lookbacks
    for lookback in lookbacks:
        # Initialize an empty list for storing the test accuracies
        accuracy_test_lknn_all = []

        # Iterate over all ks
        for k in tqdm(ks):
            X_train, Y_train, X_test, Y_test, df_test = preprocess_knn(index_df, "2013-12-31", "2018-12-31", lookback, features)
            accuracy_train_knn, accuracy_test_knn, cm_train_knn, cm_test_knn, X_train_knn, X_test_knn = knn_accuracy(X_train, Y_train, X_test, Y_test, k)
            accuracy_train_lknn, accuracy_test_lknn, cm_train_lknn, cm_test_lknn, X_train_lknn, X_test_lknn = knn_accuracy(X_train, Y_train, X_test, Y_test, k, lorentzian=True)
            accuracy_test_lknn_all.append(accuracy_test_lknn)
        
        # Save the test accuracies
        np.save(f"Result/{index_name}accuracy_test_lknn_lb{lookback}top{top}.npy", np.array(accuracy_test_lknn_all))

    # Iterate over all lookbacks
    for lookback in lookbacks:
        # Load the test accuracies
        accuracy_test_lknn_all = np.load(f"Result/{index_name}accuracy_test_lknn_lb{lookback}top{top}.npy")

        # Create a figure
        plt.figure(figsize=(10, 6))

        # Plot the test accuracy for all ks
        plt.scatter(ks, accuracy_test_lknn_all, marker="x")
        plt.plot(ks, accuracy_test_lknn_all)

        # Set the labels
        plt.xlabel(r"$k$")
        plt.ylabel("Test accuracy")

        # Set the x limit
        plt.xlim(xmin=np.min(ks)-1, xmax=np.max(ks))
        
        # Set the title
        plt.title(fr"Test accuracy of Lorentzian KNN versus $k$ with lookback={lookback}")

        # Save the plot
        plt.savefig(f"Result/Figure/{infix}accuracy_test_lknn_lb{lookback}top{top}.png", dpi=300)

        # Show the plot
        plt.show()

# Run the main function
if __name__ == "__main__":
    main()