import yfinance as yf
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Function to fetch historical stock data
def fetch_historical_data(symbol, period, interval):
    stock = yf.Ticker(symbol)
    yf_data = stock.history(period=period, interval=interval).round(2)

    data = pd.DataFrame(yf_data).reset_index()
    data.rename(columns={'Date': 'datetime'}, inplace=True)

    data = data[['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]

    return data

# Function to handle missing values
def handle_missing_values(data):
    imputer = SimpleImputer(strategy='mean')
    numeric_columns = data.select_dtypes(include=[np.number]).columns

    # Check if there are missing values in the numeric columns
    missing_columns = data[numeric_columns].columns[data[numeric_columns].isnull().any()].tolist()

    if missing_columns:
        # Fill missing values in each column separately
        for column in missing_columns:
            data[column] = imputer.fit_transform(data[column].values.reshape(-1, 1))

    return data

# Function to remove outliers
def remove_outliers(data):
    z_scores = data[['Close', 'Volume']].apply(
        lambda x: (x - x.mean()) / max(x.std(), 1e-6)
    )
    data = data[(z_scores < 3).all(axis=1)]
    return data

# Function to calculate Relative Strength Index (RSI)
def calculate_rsi(close_prices, window=14):
    delta = close_prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    relative_strength = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + relative_strength))
    return rsi

# Function to calculate Stochastic oscillator
def calculate_stochastic(close_prices, low_prices, high_prices, window=14):
    lowest_low = low_prices.rolling(window).min()
    highest_high = high_prices.rolling(window).max()
    stochastic = ((close_prices - lowest_low) / (highest_high - lowest_low)) * 100
    return stochastic

# Function to calculate Bollinger Bands
def calculate_bollinger_bands(close_prices, window=20, num_std=2):
    rolling_mean = close_prices.rolling(window=window).mean()
    rolling_std = close_prices.rolling(window=window).std()
    upper_band = rolling_mean + num_std * rolling_std
    lower_band = rolling_mean - num_std * rolling_std
    bollinger_bands = upper_band - lower_band
    return bollinger_bands

# Function to calculate Moving Average Convergence Divergence (MACD)
def calculate_macd(close_prices, window_short=12, window_long=26, window_signal=9):
    ema_short = close_prices.ewm(span=window_short).mean()
    ema_long = close_prices.ewm(span=window_long).mean()
    macd = ema_short - ema_long
    signal_line = macd.ewm(span=window_signal).mean()
    macd_data = macd - signal_line
    return macd_data

# Function to calculate technical indicators
def calculate_technical_indicators(data):
    data['MA_20'] = data['Close'].rolling(window=20).mean()
    data['MA_50'] = data['Close'].rolling(window=50).mean()
    data['RSI'] = calculate_rsi(data['Close'], window=14)
    data['Stochastic'] = calculate_stochastic(data['Close'], data['Low'], data['High'], window=14)
    data['Bollinger_Band'] = calculate_bollinger_bands(data['Close'], window=20, num_std=2)
    data['MACD'] = calculate_macd(data['Close'], window_short=12, window_long=26, window_signal=9)

    data = handle_missing_values(data)
    data = remove_outliers(data)

    # data.dropna(inplace=True)
    return data

def perform_feature_scaling(data):
    if data.empty:
        return data

    scaler = StandardScaler()
    selected_columns = ['Close', 'Volume', 'MA_20', 'MA_50', 'RSI', 'Stochastic', 'Bollinger_Band', 'MACD']

    scaler.fit(data.loc[:, selected_columns])
    # Commenting out the scaler.transform() line
    # data.loc[:, selected_columns] = scaler.transform(data.loc[:, selected_columns])

    return data

# Function to load trained models
def load_trained_models(folder_path):
    models = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.pkl'):
            model_path = os.path.join(folder_path, file_name)
            try:
                print(f"Loading model: {file_name}")
                with open(model_path, 'rb') as file:
                    model = joblib.load(file)
                print(f"Loaded model: {model}")
                models[file_name] = model
            except Exception as e:
                print(f"Error loading model {file_name}: {e}")

    return models

# Function to predict using loaded models
def predict(models, data):
    columns_to_keep = ['Close', 'Volume', 'MA_20', 'MA_50', 'RSI', 'Stochastic', 'Bollinger_Band', 'MACD']
    data = data[columns_to_keep]

    for column in data.columns:
        if data[column].dtype == 'object':
            data[column] = pd.to_numeric(data[column], errors='coerce')

    predictions = {}
    for model_name, model in models.items():
        prediction = model.predict(data)
        signal = np.where(prediction == 1, "Buy", np.where(prediction == -1, "Sell", "Hold"))
        predictions[model_name] = signal

    return predictions

# Function to generate signals based on predictions
def generate_signals(predictions):
    model_names = list(predictions.keys())
    signals = list(predictions.values())

    df_signals = pd.DataFrame(columns=model_names)
    for i in range(len(model_names)):
        df_signals[model_names[i]] = signals[i]

    aggregated_signal = np.where(
        df_signals[model_names].apply(lambda x: x == 'Buy').sum(axis=1) > len(model_names) / 2,
        'Buy',
        np.where(
            df_signals[model_names].apply(lambda x: x == 'Sell').sum(axis=1) > len(model_names) / 2,
            'Sell',
            'Hold'
        )
    )
    df_signals['Aggregated Signal'] = aggregated_signal

    return df_signals

# Function to generate the tabular format
def generate_tabular_format(data, predictions):
    model_names = list(predictions.keys())
    selected_data = data.copy()
    selected_data[model_names] = pd.DataFrame(predictions.values()).T
    return selected_data

# Function to write the tabular data to a text file
def write_to_text_file(data, predictions, output_file):
    # Create a copy of the data
    selected_data = data.copy()

    # Select the desired columns from the fetched historical data
    columns_to_select = ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
    selected_data = selected_data[columns_to_select]

    # Add columns for each model and populate with predicted values
    model_names = list(predictions.keys())
    for model_name, signal in predictions.items():
        selected_data[model_name] = signal

    # Add the Aggregated Signal column
    aggregated_signal = np.where(
        selected_data[model_names].apply(lambda x: x == 'Buy').sum(axis=1) > len(model_names) / 2,
        'Buy',
        np.where(
            selected_data[model_names].apply(lambda x: x == 'Sell').sum(axis=1) > len(model_names) / 2,
            'Sell',
            'Hold'
        )
    )
    selected_data['Aggregated Signal'] = aggregated_signal

    # Reorder the columns
    columns_to_select = ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume'] + model_names + ['Aggregated Signal']
    selected_data = selected_data[columns_to_select]

    # Write the data to a text file
    with open(output_file, 'w') as file:
        file.write(selected_data.to_string(index=False))

    print(f"Data saved to {output_file}")

# Main function
def main():
    symbol = 'INFY.NS'
    period = '1y'
    interval = '1d'
    output_file = 'Model_Output.txt'

    # Fetch historical data
    data = fetch_historical_data(symbol, period, interval)

    # Calculate technical indicators
    data = calculate_technical_indicators(data)

    # Perform feature scaling
    data = perform_feature_scaling(data)

    # Load trained models
    models_folder = 'Trained_Models'
    models = load_trained_models(models_folder)

    # Make predictions
    predictions = predict(models, data)

    # Generate signals
    signals = generate_signals(predictions)

    # Write to output file
    write_to_text_file(data, signals, output_file)


if __name__ == "__main__":
    main()
