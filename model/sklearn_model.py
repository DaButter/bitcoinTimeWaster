import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

def create_lagged_features(df, n_lags=7):
    """Create lagged features for time series prediction."""
    for i in range(1, n_lags + 1):
        df[f'Close_Lag_{i}'] = df['Close'].shift(i)
    return df

def train_model(df):
    """Train a model to predict the closing price for the next 7 days."""
    # Create lagged features
    df = create_lagged_features(df)

    # Use additional features like RSI, SMA, etc.
    # additional_features = ['SMA7', 'RSI', 'MACD', 'Upper_Band', 'Lower_Band', 'OBV']
    # additional_features = ['SMA7', 'SMA14', 'SMA21', 'EMA50', 'EMA200', 'RSI', 'EMA6', 'EMA13', 'MACD',
    #                        'Signal_Line', '10_day_SMA', 'Upper_Band', 'Lower_Band', 'ROC7', 'Stochastic14',
    #                        'ATR14', 'OBV', 'Williams_%R', 'Momentum']
    additional_features = ['SMA7', 'SMA14', 'SMA21', 'RSI', 'EMA6', 'EMA13', 'MACD',
                           'Signal_Line', '10_day_SMA', 'Upper_Band', 'Lower_Band', 'ROC7', 'Stochastic14',
                           'ATR14', 'OBV', 'Williams_%R', 'Momentum']

    # Drop rows with NaN values
    df = df.dropna()

    # Define features and target
    feature_columns = [col for col in df.columns if col.startswith('Close_Lag')] + additional_features
    features = df[feature_columns]
    target = df['Close']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=False)

    # Initialize and train the Random Forest model
    model = RandomForestRegressor(n_estimators=1000, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')

    # Prepare input data for next 7 days prediction
    last_row = df.iloc[-1]

    # Initialize the feature set for the next 7 days
    next_days_features = [last_row[f'Close_Lag_{i}'] for i in range(1, 8)]  # Lagged prices

    # Add the additional features
    for feature in additional_features:
        next_days_features.append(last_row[feature])

    predictions = []
    for i in range(7):
        # Update features for prediction
        future_features = pd.DataFrame([next_days_features], columns=feature_columns)
        future_price = model.predict(future_features)[0]
        predictions.append(future_price)

        # Shift lagged features for the next iteration
        next_days_features = [future_price] + next_days_features[:-1 - len(additional_features)]

        # Update additional features (This part can be complex depending on the feature. Use the last known value or recalculate)
        for feature in additional_features:
            next_days_features.append(last_row[feature])  # or recalculate based on the new price

    print("Predicted Closing Prices for the Next 7 Days:")
    for i, price in enumerate(predictions, start=1):
        print(f"Day {i}: ${price:.2f}")

    return model
