import pandas as pd
import numpy as np

def data_sort(csv_file):
    df = pd.read_csv(csv_file)
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%y')
    # df = df.sort_values(by='Date')
    # Set the 'Date' column as the index
    df.set_index('Date', inplace=True)
    return df

# Average True Range (ATR)
def calculate_atr(df, window=14):
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window=window).mean()
    return atr

# Stochastic Oscillator
def calculate_stochastic(df, window=14):
    low_min = df['Low'].rolling(window=window).min()
    high_max = df['High'].rolling(window=window).max()
    stoch = 100 * (df['Close'] - low_min) / (high_max - low_min)
    return stoch

def calculate_rsi(df, window=7):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Rate of Change (ROC)
def calculate_roc(df, window=7):
    roc = ((df['Close'] - df['Close'].shift(window)) / df['Close'].shift(window)) * 100
    return roc

# On-Balance Volume (OBV)
def calculate_obv(df):
    obv = pd.Series(index=df.index, dtype='float64')
    obv.iloc[0] = 0  # OBV starts from zero
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + df['Volume'].iloc[i]
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - df['Volume'].iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    return obv

# Relative Volatility Index (RVI)
def calculate_rvi(df, window=14):
    std_dev = df['Close'].rolling(window=window).std()
    rvi = 100 * (std_dev.rolling(window=window).mean() / std_dev.max())
    return rvi

# Williams %R
def calculate_williams_r(df, window=14):
    high_max = df['High'].rolling(window=window).max()
    low_min = df['Low'].rolling(window=window).min()
    williams_r = -100 * (high_max - df['Close']) / (high_max - low_min)
    return williams_r

# Commodity Channel Index (CCI)
def calculate_cci(df, window=20):
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    sma = typical_price.rolling(window=window).mean()
    mean_deviation = typical_price.rolling(window=window).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=False)
    cci = (typical_price - sma) / (0.015 * mean_deviation)
    return cci

def calculate_momentum(df, window=14):
    momentum = df['Close'] - df['Close'].shift(window)
    return momentum

def calc_statistic_params(csv_file, path_to_save="./data/BTC-USD_with_indicators.csv"):
    df = data_sort(csv_file)

    # Simple Moving Averages (SMA)
    df['SMA7'] = df['Close'].rolling(window=7).mean()
    df['SMA14'] = df['Close'].rolling(window=14).mean()
    df['SMA21'] = df['Close'].rolling(window=21).mean()

    # Exponential Moving Averages (EMA)
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()

    # Relative Strength Index (RSI)
    df['RSI'] = calculate_rsi(df)

    # Moving Average Convergence Divergence (MACD)
    df['EMA6'] = df['Close'].ewm(span=6, adjust=False).mean()
    df['EMA13'] = df['Close'].ewm(span=13, adjust=False).mean()
    df['MACD'] = df['EMA6'] - df['EMA13']
    df['Signal_Line'] = df['MACD'].ewm(span=5, adjust=False).mean()

    # Bollinger Bands
    df['10_day_SMA'] = df['Close'].rolling(window=10).mean()
    df['10_day_std'] = df['Close'].rolling(window=10).std()
    df['Upper_Band'] = df['10_day_SMA'] + (df['10_day_std'] * 2)
    df['Lower_Band'] = df['10_day_SMA'] - (df['10_day_std'] * 2)

    # Rate of Change (ROC)
    df['ROC7'] = calculate_roc(df, window=7)
    df['ROC14'] = calculate_roc(df, window=14)

    # Stochastic Oscillator
    df['Stochastic14'] = calculate_stochastic(df)

    # Average True Range (ATR)
    df['ATR14'] = calculate_atr(df)

    # On-Balance Volume (OBV)
    df['OBV'] = calculate_obv(df)

    # Williams %R
    df['Williams_%R'] = calculate_williams_r(df)

    # Commodity Channel Index (CCI)
    df['CCI'] = calculate_cci(df)

    # Relative Volatility Index (RVI)
    df['RVI'] = calculate_rvi(df)

    # Momentum
    df['Momentum'] = calculate_momentum(df)

    # Show the first few rows of the DataFrame with the new columns
    print(df.head())

    # Save the enhanced DataFrame to a new CSV file
    df.to_csv(path_to_save)
    print(f"CSV file with statistics indicators saved in '{path_to_save}'...")
    return df


if __name__ == "__main__":
    calc_statistic_params('./data/BTC-USD_full_copy.csv', '../data/BTC-USD_with_indicators.csv')
