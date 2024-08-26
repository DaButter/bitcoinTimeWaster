import pandas as pd
import matplotlib.pyplot as plt

def full_btc_trendline(csv_file):
    # Load the CSV file from the data directory
    df = pd.read_csv(csv_file)

    # Convert the 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Ensure the data is sorted by date
    df = df.sort_values(by='Date')

    # Plot the trendline of 'Close' prices over time
    plt.figure(figsize=(10, 6))
    plt.plot(df['Date'], df['Close'], label='Close Price', color='blue')

    # Set plot title and labels
    plt.title('Bitcoin Price Trendline')
    plt.xlabel('Date')
    plt.ylabel('Close Price (USD)')

    # Rotate date labels for better readability
    plt.xticks(rotation=45)

    # Display the legend
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    full_btc_trendline('./data/BTC-USD_full_copy.csv')