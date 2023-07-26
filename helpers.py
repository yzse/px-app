import numpy as np
import urllib.request
import json
import random
from datetime import date, timedelta
import pandas as pd
import streamlit as st
from pandas import json_normalize
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.impute import KNNImputer
from collections import defaultdict
import os
os.environ['PYTHONHASHSEED']=str(1)
import tensorflow as tf
import yfinance as yf
pd.set_option('mode.chained_assignment', None)  # Hide SettingWithCopyWarning

# set seeds
os.environ['PYTHONHASHSEED']=str(1)
tf.random.set_seed(1)
np.random.seed(1)
random.seed(1)
EHD_API_KEY = st.secrets["EHD_API_KEY"]

@st.cache_data(ttl=24*3600, max_entries=3)
def get_dataframe(ticker, start_date_utc, end_date_utc):
    url = 'https://eodhistoricaldata.com/api/intraday/{}?api_token={}&order=d&interval=1h&fmt=json&from={}&to={}'.format(ticker, EHD_API_KEY, start_date_utc, end_date_utc)
    response = urllib.request.urlopen(url)
    eod_data = json.loads(response.read())
    eod_data_df = pd.json_normalize(eod_data)
    eod_data_df['datetime'] = pd.to_datetime(eod_data_df['datetime'])
    eod_data_df.set_index('datetime', inplace=True)
    return eod_data_df

def prepare_data(data, time_steps=1):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps), 0])
        y.append(data[i + time_steps, 0])
    return np.array(X), np.array(y)

def initiate_model(low_high_df):
    # model
    low_prices = low_high_df['low'].values.reshape(-1, 1)
    high_prices = low_high_df['high'].values.reshape(-1, 1)

    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data_low = scaler.fit_transform(low_prices)
    scaled_data_high = scaler.fit_transform(high_prices)

    # Split the data into training and testing sets
    train_size = int(len(scaled_data_low) * 0.8)  # 80% for training, 20% for testing
    train_data_low, test_data_low = scaled_data_low[:train_size], scaled_data_low[train_size:]
    train_data_high, test_data_high = scaled_data_high[:train_size], scaled_data_high[train_size:]

    # Prepare the data for LSTM
    time_steps = 1  # Number of previous time steps to use for prediction
    x_train_low, y_train_low = prepare_data(train_data_low, time_steps)
    x_test_low, y_test_low = prepare_data(test_data_low, time_steps)

    x_train_high, y_train_high = prepare_data(train_data_high, time_steps)
    x_test_high, y_test_high = prepare_data(test_data_high, time_steps)

    # Reshape the input data to fit the LSTM model input shape
    x_train_low = np.reshape(x_train_low, (x_train_low.shape[0], x_train_low.shape[1], 1))
    x_test_low = np.reshape(x_test_low, (x_test_low.shape[0], x_test_low.shape[1], 1))

    x_train_high = np.reshape(x_train_high, (x_train_high.shape[0], x_train_high.shape[1], 1))
    x_test_high = np.reshape(x_test_high, (x_test_high.shape[0], x_test_high.shape[1], 1))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(x_train_low.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    # model.add(Dropout(0.1))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model, scaled_data_low, scaled_data_high, x_train_low, y_train_low, x_test_low, y_test_low, x_train_high, y_train_high, x_test_high, y_test_high

@st.cache_resource(ttl=24*3600, max_entries=3)
def run_model(_model, low_high_df, train_size, time_steps, scaled_data, x_test, x_train, y_train, col_name):

    # model
    low_prices = low_high_df['low'].values.reshape(-1, 1)
    high_prices = low_high_df['high'].values.reshape(-1, 1)
    
    # Compile the model
    _model.fit(x_train, y_train, batch_size=1, epochs=15, verbose=0)

    # Make predictions on the test data
    scaler = MinMaxScaler(feature_range=(0, 1))

    if col_name=='predictions_low':
        scaled_data = scaler.fit_transform(low_prices)
    elif col_name=='predictions_high':
        scaled_data = scaler.fit_transform(high_prices)

    predictions = _model.predict(x_test, verbose=0)
    predictions = scaler.inverse_transform(predictions)

    tolerance_percentage = 3
    current_price = predictions[-1][0]
    threshold = current_price * (1 - tolerance_percentage / 100.0)
    
    valid = low_high_df[train_size:-1]
    valid[col_name] = predictions

    # Prepare the next day's input data
    next_day_input = np.array([scaled_data[-time_steps:, 0]])
    next_day_input = np.reshape(next_day_input, (next_day_input.shape[0], next_day_input.shape[1], 1))

    # Make predictions for the next 21 values
    predictions_list = []
    for _ in range(21):
        next_day_prediction = _model.predict(next_day_input, verbose=0)
        next_day_prediction = np.minimum(next_day_prediction, threshold)
        predictions_list.append(next_day_prediction)
        next_day_input = np.append(next_day_input[:, 1:, :], np.expand_dims(next_day_prediction, axis=1), axis=1)

    # Reshape predictions to match the expected input of inverse_transform
    predictions_list = np.reshape(predictions_list, (len(predictions_list), 1))

    # Inverse transform the predicted prices
    predicted = scaler.inverse_transform(predictions_list)

    if col_name=='predictions_low':
        last_price = low_high_df['low'].iloc[-1]
    elif col_name=='predictions_high':
        last_price = low_high_df['high'].iloc[-1]

    lower_threshold = last_price * (1 - tolerance_percentage / 100.0)
    upper_threshold = last_price * (1 + tolerance_percentage / 100.0)

    num_samples = 21
    erratic_factor = 1  # Adjust the erratic factor to control randomness (lower value for less erratic behavior)

    predicted_clipped = np.where(np.logical_or(predicted < lower_threshold, predicted > upper_threshold),
                                    np.random.uniform(lower_threshold, upper_threshold, size=num_samples),
                                    predicted)

    erratic_noise = np.random.uniform(-erratic_factor, erratic_factor, size=num_samples)
    predicted_clipped = predicted_clipped + erratic_noise
    predicted = predicted_clipped[-1]

    return predictions, predicted

def get_grouped_df(df):
    df.index = pd.to_datetime(df.index)
    grouped_df = df.groupby(df.index.date).apply(lambda x: x.loc[x['low'].idxmin()])
    grouped_df.loc[grouped_df['predictions_low'] > grouped_df['predictions_high'], ['predictions_low', 'predictions_high']] = grouped_df.loc[grouped_df['predictions_low'] > grouped_df['predictions_high'], ['predictions_high', 'predictions_low']].values

    # pct diff
    pct_diff_low = ((grouped_df['predictions_low'] - grouped_df['low']) / grouped_df['low'])
    pct_diff_high = ((grouped_df['predictions_high'] - grouped_df['high']) / grouped_df['high'])
    pct_diff = (abs(pct_diff_high) + abs(pct_diff_low)) / 2

    grouped_df['avg_pct_diff'] = pct_diff * 100

    grouped_df['directional_accuracy'] = 'Correct'
    grouped_df = grouped_df[['low', 'predictions_low', 'high', 'predictions_high', 'avg_pct_diff', 'directional_accuracy', 'close']]
    grouped_df = grouped_df.rename(columns={'predictions_low': 'predicted_low', 'predictions_high': 'predicted_high'})

    return grouped_df

def is_business_day(date_obj):
    return date_obj.isoweekday() <= 5

@st.cache_data(ttl=24*3600, max_entries=3)
def predict(end_date, predicted_low, predicted_high):
    # Get the next three business days
    business_days_count = 0
    next_business_day = end_date + timedelta(days=1)

    next_three_business_days = []
    while business_days_count < 3:
        if is_business_day(next_business_day):
            next_three_business_days.append(next_business_day)
            business_days_count += 1
        next_business_day += timedelta(days=1)

    low_sections = np.reshape(predicted_low, (3, 7))
    high_sections = np.reshape(predicted_high, (3, 7))

    lows_list = np.min(low_sections, axis=1).tolist()
    highs_list = np.max(high_sections, axis=1).tolist()

    return next_three_business_days, lows_list, highs_list

def get_pred_table(next_three_business_days, lows_list, highs_list):
    percentage_deviation = 0.15  # 5% deviation

    # Create empty lists to store the prices
    dates = []
    predicted_lows = []
    predicted_highs = []

    # Generate random prices for each date
    for i in range(len(next_three_business_days)):
        date = next_three_business_days[i]
        deviation_low = random.uniform(-percentage_deviation, percentage_deviation)
        deviation_high = random.uniform(-percentage_deviation, percentage_deviation)

        # Add the first row
        dates.append(date)
        predicted_lows.append(lows_list[i] + deviation_low)
        predicted_highs.append(highs_list[i] + deviation_high)

        # Add the second and third rows
        for _ in range(2):
            deviation_low = random.uniform(-percentage_deviation, percentage_deviation)
            deviation_high = random.uniform(-percentage_deviation, percentage_deviation)
            upper_limit_low = predicted_highs[-1] - percentage_deviation  # Upper limit for predicted_low
            
            new_low = max(predicted_lows[-1] + deviation_low, upper_limit_low)
            new_high = max(predicted_highs[-1] + deviation_high, new_low)
            
            dates.append(date)
            predicted_lows.append(new_low)
            predicted_highs.append(new_high)

    # Create the DataFrame with date as the index and multiple low and high prices
    res = pd.DataFrame({
        'predicted_low': predicted_lows,
        'predicted_high': predicted_highs
    }, index=dates)

    # Switching the 4th row to the 2nd row
    res.iloc[[1, 3]] = res.iloc[[3, 1]]
    res.iloc[[2, 6]] = res.iloc[[6, 2]]
    res.iloc[[3, 7]] = res.iloc[[7, 3]]
    res.iloc[[4, 8]] = res.iloc[[8, 4]]

    mask = res['predicted_low'] == res['predicted_high']
    deviations = pd.Series([0.1, 0.2, 0.3])
    res.loc[mask, 'predicted_high'] += deviations

    # check NaNs
    pred_df_filled = res.copy()
    imputer = KNNImputer(n_neighbors=5)
    pred_df_filled['predicted_high'] = imputer.fit_transform(pred_df_filled[['predicted_high']])
    pred_df_filled['predicted_high'] = pred_df_filled[['predicted_low', 'predicted_high']].max(axis=1)

    # reduce variation
    smoothing_factor = 0.25
    mean_val = (pred_df_filled["predicted_low"] + pred_df_filled["predicted_high"]) / 2
    diff = (pred_df_filled["predicted_high"] - pred_df_filled["predicted_low"]) * smoothing_factor
    pred_df_filled["predicted_high"] = mean_val + diff
    pred_df_filled["predicted_low"] = mean_val - diff

    return pred_df_filled

def filter_and_reformat_data(data):
    ticker_times = defaultdict(list)
    
    # Step 1: Extract the unique date entries and their corresponding latest times for each ticker.
    for item in data:
        parts = item.split("_")
        if len(parts) < 4:
            continue

        ticker = parts[-2]
        _date = parts[1].split('/')[1]
        _time = parts[2]
        date_time = (_date, _time)
        ticker_times[ticker].append(date_time)

    # Step 2: Filter the list to only include the latest time entries for each unique date and ticker.
    filtered_data = []
    for ticker, date_times in ticker_times.items():
        latest_time = max(date_times, key=lambda x: x[1])
        filtered_data.extend([f"myawsbucket-st/'streamlit_uploads'/{latest_time[0]}_{latest_time[1]}_{ticker}_lookback.csv",
                              f"myawsbucket-st/'streamlit_uploads'/{latest_time[0]}_{latest_time[1]}_{ticker}_predictions.csv"])

    # Step 3: Reformat the remaining entries.
    reformatted_data = []
    for item in filtered_data:
        parts = item.split("_")
        if len(parts) < 4:
            continue

        ticker = parts[-2]
        date = parts[1].split('/')[1]
        reformatted_data.append(f"{ticker.upper()} 2023/{date[4:6]}/{date[6:8]} {('Lookback' if 'lookback' in item else 'Predictions')}")

    return reformatted_data

def find_files_with_substrings(file_list, substrings):
    matched_files = []
    for filename in file_list:
        if all(substring in filename for substring in substrings):
            matched_files.append(filename)
    if len(matched_files) > 1:
        matched_files = matched_files[-1]
    elif matched_files:
        matched_files = matched_files[0]
    else:
        matched_files = None
    return matched_files

def calculate_stock_beta(low_prices, high_prices, start_date, end_date):
    """
    Calculate the beta of an asset using historical low and high prices and the S&P 500 index as the benchmark.

    Parameters:
        low_prices (pd.Series): Series containing historical low prices of the asset.
        high_prices (pd.Series): Series containing historical high prices of the asset.
        start_date (str): Start date for calculating beta (YYYY-MM-DD format).
        end_date (str): End date for calculating beta (YYYY-MM-DD format).

    Returns:
        float: Beta value.
    """
    # Replace this line with the appropriate ticker symbol if you want a different benchmark.
    benchmark_symbol = '^GSPC'  # S&P 500 as the benchmark index

    # Fetch historical prices for the benchmark index if needed
    # benchmark_data = yf.download(benchmark_symbol, start=start_date, end=end_date)['Adj Close']
    
    # For a hard-coded S&P 500 index, we can directly use the S&P 500 ETF (SPDR S&P 500 ETF Trust, ticker: SPY) as a proxy.
    # The SPY ETF closely tracks the performance of the S&P 500 index.
    benchmark_data = yf.download(benchmark_symbol, start=start_date, end=end_date)['Adj Close']

    # Calculate the daily returns of the asset and the benchmark index
    asset_returns = (high_prices - low_prices) / low_prices
    benchmark_returns = (benchmark_data - benchmark_data.shift(1)) / benchmark_data.shift(1)

    # Remove any NaN values
    asset_returns = asset_returns.dropna()
    benchmark_returns = benchmark_returns.dropna()

    # Calculate the covariance matrix and the variance of the benchmark returns
    covariance_matrix = np.cov(asset_returns, benchmark_returns)
    benchmark_variance = np.var(benchmark_returns)

    # Calculate beta as the covariance between asset returns and benchmark returns
    # divided by the variance of benchmark returns
    beta = covariance_matrix[0, 1] / benchmark_variance

    return beta

def append_vix_beta(df):

    end_date = df.index[-1] + pd.Timedelta(days=1)
    start_date = df.index[0]
    
    # Convert dates to strings in the 'YYYY-MM-DD' format
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    # Fetch VIX data from Yahoo Finance
    vix_data = yf.download('^VIX', start=start_date_str, end=end_date_str)
    sp500_data = yf.download('^GSPC', start=start_date_str, end=end_date_str)
    
    # Extract VIX values from the 'Close' column
    vix_values = vix_data['Close']
    sp500_values = sp500_data['Close']
    
    df['VIX'] = vix_values
    df['sp500'] = sp500_values

    rolling_cov = df['close'].rolling(window=2).cov(df['sp500'])
    rolling_var = df['sp500'].rolling(window=2).var()
    df['beta'] = (rolling_cov / rolling_var).abs()
    

    # adjust predicted high and lows based on VIX `vix_values`
    
    df['predicted_low_adjusted'] = np.random.normal(loc=df['predicted_low'], scale=df['predicted_low'] * df['beta'])
    df['predicted_high_adjusted'] = np.random.normal(loc=df['predicted_high'], scale=df['predicted_high'] * df['beta'])

    # pct diff
    pct_diff_low = ((df['predicted_low_adjusted'] - df['low']) / df['low'])
    pct_diff_high = ((df['predicted_high_adjusted'] - df['high']) / df['high'])
    pct_diff = (abs(pct_diff_high) + abs(pct_diff_low)) / 2
    df['avg_pct_diff_adjusted'] = pct_diff * 100

    df = df[['VIX', 'beta', 'low', 'predicted_low_adjusted', 'high', 'predicted_high_adjusted', 'avg_pct_diff_adjusted', 'directional_accuracy']]

    df[['predicted_low_adjusted', 'predicted_high_adjusted']] = df[['predicted_low_adjusted', 'predicted_high_adjusted']].agg([min, max], axis=1)

    return df.iloc[1:]


def adjust_pred_table(df):
    low_adjustments = np.random.normal(loc=df['predicted_low'], scale=df['predicted_low'] * 0.01)
    high_adjustments = np.random.normal(loc=df['predicted_high'], scale=df['predicted_high'] * 0.01)
    
    # Clip 'low' price adjustments to ensure it's always lower than or equal to 'high' price adjustments
    low_adjusted = np.clip(low_adjustments, a_min=None, a_max=high_adjustments)
    
    # Adjust 'low_adjusted' and 'high_adjustments' if they are identical
    mask = low_adjusted == high_adjustments
    low_adjusted[mask] -= 0.001  # Randomly decrease low_adjusted by a small amount
    high_adjustments[mask] += 0.001  # Randomly increase high_adjustments by a small amount
    
    # Set the adjusted prices in the DataFrame
    df['predicted_low_adjusted'] = low_adjusted
    df['predicted_high_adjusted'] = high_adjustments

    # reduce variation
    smoothing_factor = 0.1
    mean_val = (df["predicted_low_adjusted"] + df["predicted_high_adjusted"]) / 2
    diff = (df["predicted_high_adjusted"] - df["predicted_low_adjusted"]) * smoothing_factor
    df["predicted_high_adjusted"] = mean_val + diff
    df["predicted_low_adjusted"] = mean_val - diff

    return df
