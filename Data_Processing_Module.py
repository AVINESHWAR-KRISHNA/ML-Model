import pandas as pd
import numpy as np
from scipy.stats import norm
from Data_Collection_Module import fetch_historical_data

def preprocess_data(symbol, period, interval):
    # Fetch historical stock data
    data, columns = fetch_historical_data(symbol, period, interval)

    # Perform data preprocessing tasks
    data = handle_missing_values(data)
    data = remove_outliers(data)
    data = calculate_technical_indicators(data)
    data = generate_signals(data)

    return data


def handle_missing_values(data):
    # Handle missing values in the data (e.g., fill with mean, forward/backward fill, etc.)
    data = data.fillna(method='ffill')  # Forward fill missing values

    return data


def remove_outliers(data):
    # Remove outliers from the data (e.g., using statistical techniques or domain-specific methods)
    z_scores = (data - data.mean()) / data.std()
    data = data[(z_scores < 3).all(axis=1)]  # Keep rows with z-scores within 3 standard deviations

    return data


def calculate_technical_indicators(data):
    # Calculate technical indicators
    data['MA_20'] = data['Close'].rolling(window=20).mean()
    data['MA_50'] = data['Close'].rolling(window=50).mean()
    data['RSI'] = calculate_rsi(data['Close'], window=14)
    data['Stochastic'] = calculate_stochastic(data['Close'], data['Low'], data['High'], window=14)
    data['Bollinger_Band'] = calculate_bollinger_bands(data['Close'], window=20, num_std=2)
    data['MACD'] = calculate_macd(data['Close'], window_short=12, window_long=26, window_signal=9)

    return data


def calculate_rsi(close_prices, window=14):
    # Calculate Relative Strength Index (RSI)
    delta = close_prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    relative_strength = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + relative_strength))
    return rsi


def calculate_stochastic(close_prices, low_prices, high_prices, window=14):
    # Calculate Stochastic oscillator
    lowest_low = low_prices.rolling(window).min()
    highest_high = high_prices.rolling(window).max()
    stochastic = ((close_prices - lowest_low) / (highest_high - lowest_low)) * 100
    return stochastic


def calculate_bollinger_bands(close_prices, window=20, num_std=2):
    # Calculate Bollinger Bands
    rolling_mean = close_prices.rolling(window=window).mean()
    rolling_std = close_prices.rolling(window=window).std()
    upper_band = rolling_mean + num_std * rolling_std
    lower_band = rolling_mean - num_std * rolling_std
    bollinger_bands = upper_band - lower_band
    return bollinger_bands


def calculate_macd(close_prices, window_short=12, window_long=26, window_signal=9):
    # Calculate Moving Average Convergence Divergence (MACD)
    ema_short = close_prices.ewm(span=window_short).mean()
    ema_long = close_prices.ewm(span=window_long).mean()
    macd = ema_short - ema_long
    signal_line = macd.ewm(span=window_signal).mean()
    macd_data = macd - signal_line
    return macd_data


def calculate_probability(values):
    # Calculate probability based on values (e.g., using statistical distribution)
    mean = np.mean(values)
    std_dev = np.std(values)
    z_scores = (values - mean) / std_dev
    probabilities = norm.cdf(z_scores)
    return probabilities

def calculate_dynamic_threshold(indicator_values):
    # Calculate dynamic threshold based on indicator values
    # You can implement your own logic here based on historical data or statistical methods
    threshold = indicator_values.mean()
    return threshold

def calculate_trend_direction(close_prices, ma_window_short=20, ma_window_long=50):
    # Calculate trend direction based on moving averages
    ma_short = close_prices.rolling(ma_window_short).mean()
    ma_long = close_prices.rolling(ma_window_long).mean()
    trend_direction = np.sign(ma_short - ma_long)
    return trend_direction


def generate_signals(data):
    # Generate buy/sell signals based on technical indicators and advanced techniques
    data['Signal'] = 'Hold'

    # Generate signals based on technical indicators
    rsi = calculate_rsi(data['Close'])
    stochastic = stochastic = calculate_stochastic(data['Close'], data['Low'], data['High'])
    ma_cross = data['MA_20'] > data['MA_50']

    # Generate Bollinger Bands
    data['MA_20'] = data['Close'].rolling(window=20).mean()
    data['std_dev'] = data['Close'].rolling(window=20).std()
    data['upper_band'] = data['MA_20'] + (2 * data['std_dev'])
    data['lower_band'] = data['MA_20'] - (2 * data['std_dev'])

    # Calculate probabilities based on Bollinger Bands
    upper_band_probabilities = calculate_probability(data['upper_band'])
    lower_band_probabilities = calculate_probability(data['lower_band'])

    # Generate signals based on dynamic thresholds
    rsi_threshold = calculate_dynamic_threshold(rsi)
    stochastic_threshold = calculate_dynamic_threshold(stochastic)

    # Signal Confirmation: Require multiple indicators to align for signal generation
    signal_confirmation = (
        (rsi < rsi_threshold) &
        (stochastic < stochastic_threshold) &
        ma_cross
    )

    # Trend Analysis: Generate signals in the direction of the prevailing trend
    trend_direction = calculate_trend_direction(data['Close'])
    trend_confirmation = (trend_direction > 0)

    # Combine signals based on confirmation and probabilities
    buy_condition = (
        signal_confirmation &
        trend_confirmation &
        (upper_band_probabilities > 0.8) &
        (lower_band_probabilities > 0.8)
    )
    sell_condition = (
        signal_confirmation &
        trend_confirmation &
        (upper_band_probabilities < 0.2) &
        (lower_band_probabilities < 0.2)
    )

    data.loc[buy_condition, 'Signal'] = 'Buy'
    data.loc[sell_condition, 'Signal'] = 'Sell'

    # Add signal reasons
    data['Signal_Reason'] = np.where(
        data['Signal'] == 'Buy',
        'Buy signal confirmed based on multiple indicators and trend analysis',
        np.where(
            data['Signal'] == 'Sell',
            'Sell signal confirmed based on multiple indicators and trend analysis',
            'No clear signal'
        )
    )

    # Additional risk management techniques
    stop_loss_pct = 0.03  # 3% stop-loss
    take_profit_pct = 0.05  # 5% take-profit
    initial_capital = 100000  # Initial capital for position sizing

    # Calculate position size based on available capital and stop-loss percentage
    data['Position_Size'] = initial_capital * stop_loss_pct / (data['Close'] * (1 + stop_loss_pct))

    # Calculate trailing stop-loss and take-profit levels
    data['Trailing_Stop_Loss'] = data['Close'] * (1 - stop_loss_pct)
    data['Take_Profit'] = data['Close'] * (1 + take_profit_pct)

    return data

# Example usage:

# preprocessed_data = preprocess_data(Info_Ticker.stock_symbol, Info_Ticker.start_date, Info_Ticker.end_date, Info_Ticker.time_frame)