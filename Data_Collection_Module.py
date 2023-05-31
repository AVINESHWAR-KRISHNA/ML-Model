import pandas as pd
import yfinance as yf
import Info_Ticker

def fetch_historical_data(symbol, period, interval):
    try:
        # Fetch data from Yahoo Finance
        
        stock = yf.Ticker(symbol)
        yf_data = stock.history(period=period, interval=interval)
        
        # Convert Yahoo Finance data to a Pandas DataFrame
        data = pd.DataFrame(yf_data).reset_index(drop=True)

        data = data[['Open','High','Low','Close','Volume']]
        # Set column names for future reference

        columns = list(data.columns)
        
        # Validate data
        validate_data(data)
        
        print(data.head(1))
        return data, columns
    
    except Exception as e:
        raise ValueError(f"Error fetching historical data: {str(e)}")


def validate_data(df):
    # Perform data validation checks here
    # Check for missing values, anomalies, or unexpected patterns
    # Raise appropriate exceptions if any issues are found
    
    # Example: Check for missing values
    if df.isnull().values.any():
        raise ValueError("Missing values found in the data.")
    
# fetch_historical_data(Info_Ticker.symbol, Info_Ticker.period, Info_Ticker.interval)
