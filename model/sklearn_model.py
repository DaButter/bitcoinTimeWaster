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

    # Drop rows with NaN values
    df = df.dropna()

    # Define features and target
    feature_columns = [col for col in df.columns if col.startswith('Close_Lag')]
    features = df[feature_columns]
    target = df['Close']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=False)

    # Initialize and train the Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
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
    next_days_features = [last_row[f'Close_Lag_{i}'] for i in range(1, 8)]

    predictions = []
    for i in range(7):
        # Update features for prediction
        future_features = pd.DataFrame([next_days_features], columns=feature_columns)
        future_price = model.predict(future_features)[0]
        predictions.append(future_price)
        # Update features for next iteration
        next_days_features = [future_price] + next_days_features[:-1]

    print("Predicted Closing Prices for the Next 7 Days:")
    for i, price in enumerate(predictions, start=1):
        print(f"Day {i}: ${price:.2f}")

    return model

