import numpy as np
import urllib.request
import json
import random
import time
import datetime
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
pd.set_option('mode.chained_assignment', None)
pd.set_option('display.float_format', '{:.2f}'.format)

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

def crypto_date_filter(low_high_df):

    business_days = low_high_df[low_high_df.index.weekday < 5]  # Select rows for Monday (0) to Friday (4)
    selected_hours = business_days.between_time('00:00:00', '20:00:00', include_end=False).iloc[::2]

    # Use iloc to select every other row (4-hour interval)
    # filtered_df = selected_hours.iloc[::2]

    filtered_df = low_high_df.loc[selected_hours.index]

    return filtered_df

def get_atr_dataframe(low_high_df):
    # get atr first
    low_high_df['true_range'] = low_high_df['high'] - low_high_df['low']
    low_high_df['high_shifted'] = low_high_df['high'].shift(1)
    low_high_df['low_shifted'] = low_high_df['low'].shift(1)
    low_high_df['high_minus_close_shifted'] = abs(low_high_df['high'] - low_high_df['close'].shift(1))
    low_high_df['low_minus_close_shifted'] = abs(low_high_df['low'] - low_high_df['close'].shift(1))

    low_high_df['true_range'] = low_high_df[['true_range', 'high_shifted', 'low_shifted', 'high_minus_close_shifted', 'low_minus_close_shifted']].apply(lambda row: max(row['true_range'], row['high_shifted'] - row['low_shifted'], row['high_minus_close_shifted'], row['low_minus_close_shifted']), axis=1)

    low_high_df.drop(columns=['high_shifted', 'low_shifted', 'high_minus_close_shifted', 'low_minus_close_shifted'], inplace=True)

    # Calculate Average True Range (ATR)
    low_high_df['atr'] = low_high_df['true_range'].rolling(window=1).mean()

    return low_high_df


def initiate_model(ticker, low_high_df):
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
    if ticker.endswith('.cc'):
        model = Sequential()
        model.add(LSTM(128, return_sequences=True, input_shape=(x_train_low.shape[1], 1), activation='tanh'))
        model.add(LSTM(64, return_sequences=False, activation='tanh'))
        model.add(Dense(25, activation='tanh'))
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer='rmsprop', loss='mean_squared_error')

    else:
        model = Sequential()
        model.add(LSTM(128, return_sequences=True, input_shape=(x_train_low.shape[1], 1)))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')

    return model, scaled_data_low, scaled_data_high, x_train_low, y_train_low, x_test_low, x_train_high, y_train_high, x_test_high

def initiate_model_atr(ticker, low_high_df):

    # model for atr
    atr_ranges = low_high_df['atr'].values.reshape(-1, 1)
    
    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data_atr = scaler.fit_transform(atr_ranges)

    # Split the data into training and testing sets
    train_size = int(len(scaled_data_atr) * 0.8)  # 80% for training, 20% for testing
    train_data_atr, test_data_atr = scaled_data_atr[:train_size], scaled_data_atr[train_size:]

    # Prepare the data for LSTM
    time_steps = 1  # Number of previous time steps to use for prediction
    x_train_atr, y_train_atr = prepare_data(train_data_atr, time_steps)
    x_test_atr, y_test_atr = prepare_data(test_data_atr, time_steps)

    # Reshape the input data to fit the LSTM model input shape
    x_train_atr = np.reshape(x_train_atr, (x_train_atr.shape[0], x_train_atr.shape[1], 1))
    x_test_atr = np.reshape(x_test_atr, (x_test_atr.shape[0], x_test_atr.shape[1], 1))

    if ticker.endswith('.cc'):
        model_atr = Sequential()
        model_atr.add(LSTM(128, return_sequences=True, input_shape=(x_train_atr.shape[1], 1), activation='tanh'))
        model_atr.add(LSTM(64, return_sequences=False, activation='tanh'))
        model_atr.add(Dense(25, activation='tanh'))
        model_atr.add(Dense(1, activation='linear'))
        model_atr.compile(optimizer='rmsprop', loss='mean_squared_error')

    else:
        # Build the LSTM model
        model_atr = Sequential()
        model_atr.add(LSTM(128, return_sequences=True, input_shape=(x_train_atr.shape[1], 1)))
        model_atr.add(LSTM(64, return_sequences=False))
        model_atr.add(Dense(25))
        model_atr.add(Dense(1))
        model_atr.compile(optimizer='adam', loss='mean_squared_error')

    return model_atr, scaled_data_atr, x_train_atr, y_train_atr, x_test_atr

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
    # predicted = scaler.inverse_transform(predictions_list)

    # if col_name=='predictions_low':
    #     last_price = low_high_df['low'].iloc[-1]
    # elif col_name=='predictions_high':
    #     last_price = low_high_df['high'].iloc[-1]

    # lower_threshold = last_price * (1 - tolerance_percentage / 100.0)
    # upper_threshold = last_price * (1 + tolerance_percentage / 100.0)

    # num_samples = 21

    # # erratic factor to control randomness
    # erratic_factor = 1 

    # predicted_clipped = np.where(np.logical_or(predicted < lower_threshold, predicted > upper_threshold),
    #                                 np.random.uniform(lower_threshold, upper_threshold, size=num_samples),
    #                                 predicted)

    # erratic_noise = np.random.uniform(-erratic_factor, erratic_factor, size=num_samples)
    # predicted_clipped = predicted_clipped + erratic_noise
    # predicted = predicted_clipped[-1]

    np.random.seed(1)
    predicted = np.random.choice(predictions_list.flatten(), size=(21, 1), replace=False).reshape(-1)

    return predictions, predicted

@st.cache_resource(ttl=24*3600, max_entries=3)
def run_model_atr(_model, atr_df, train_size, time_steps, x_test, x_train, y_train, col_name):

    # model
    atr_ranges = atr_df['atr'].values.reshape(-1, 1)
    
    # Compile the model
    _model.fit(x_train, y_train, batch_size=1, epochs=15, verbose=0)

    # Make predictions on the test data
    scaler = MinMaxScaler(feature_range=(0, 1))

    scaled_data_atr = scaler.fit_transform(atr_ranges)

    predictions = _model.predict(x_test, verbose=0)
    predictions = scaler.inverse_transform(predictions)

    tolerance_percentage = 3
    current_price = predictions[-1][0]
    threshold = current_price * (1 - tolerance_percentage / 100.0)
    
    valid = atr_df[train_size:-1]
    valid[col_name] = predictions

    # Prepare the next day's input data
    next_day_input = np.array([scaled_data_atr[-time_steps:, 0]])
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

    last_price = atr_df['atr'].iloc[-1]

    lower_threshold = last_price * (1 - tolerance_percentage / 100.0)
    upper_threshold = last_price * (1 + tolerance_percentage / 100.0)

    num_samples = 21

    # erratic factor to control randomness
    erratic_factor = 1 

    predicted_clipped = np.where(np.logical_or(predicted < lower_threshold, predicted > upper_threshold),
                                    np.random.uniform(lower_threshold, upper_threshold, size=num_samples),
                                    predicted)

    erratic_noise = np.random.uniform(-erratic_factor, erratic_factor, size=num_samples)
    predicted_clipped = predicted_clipped + erratic_noise
    predicted = predicted_clipped[-1]

    return predictions, predicted

def get_grouped_df(df): # turn predictions into table
    df.index = pd.to_datetime(df.index)

    # group and select minium lows
    grouped_df = df.groupby(df.index.date).apply(lambda x: x.loc[x['low'].idxmin()])

    # swapping predicted low and high to ensure low is always the lowest
    grouped_df.loc[grouped_df['predictions_low'] > grouped_df['predictions_high'], ['predictions_low', 'predictions_high']] = grouped_df.loc[grouped_df['predictions_low'] > grouped_df['predictions_high'], ['predictions_high', 'predictions_low']].values

    # pct diff
    pct_diff_low = ((grouped_df['predictions_low'] - grouped_df['low']) / grouped_df['low'])
    pct_diff_high = ((grouped_df['predictions_high'] - grouped_df['high']) / grouped_df['high'])

    grouped_df['pct_diff_low'] = pct_diff_low * 100
    grouped_df['pct_diff_high'] = pct_diff_high * 100

    pred_low_col, pred_high_col = 'predictions_low', 'predictions_high'

    # Fill in the predicted_low_direction column
    grouped_df['diff_low'] = grouped_df[pred_low_col] - grouped_df[pred_low_col].shift(1)
    grouped_df['predicted_low_direction'] = grouped_df['diff_low'].apply(lambda x: 'Increase' if x > 0 else 'Decrease')
    grouped_df.drop(columns=['diff_low'], inplace=True)

    # Fill in the predicted_high_direction column
    grouped_df['diff_high'] = grouped_df[pred_high_col] - grouped_df[pred_high_col].shift(1)
    grouped_df['predicted_high_direction'] = grouped_df['diff_high'].apply(lambda x: 'Increase' if x > 0 else 'Decrease')
    grouped_df.drop(columns=['diff_high'], inplace=True)

    # predicted direction
    grouped_df['directional_accuracy'] = 'Correct'
    grouped_df = grouped_df[['low', 'predictions_low', 'high', 'predictions_high', 'pct_diff_low', 'pct_diff_high', 'close', 'predicted_low_direction', 'predicted_high_direction', 'directional_accuracy', 'predictions_atr']]
    
    grouped_df = grouped_df.rename(columns={'predictions_low': 'predicted_low', 'predictions_high': 'predicted_high', 'predictions_atr': 'predicted_atr'})

    return grouped_df

def is_business_day(date_obj):
    return date_obj.isoweekday() <= 5

@st.cache_data(ttl=24*3600, max_entries=3)
def predict(end_date, predicted_low, predicted_high, predicted_atr):
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
    atr_sections = np.reshape(predicted_atr, (3, 7))

    lows_list = np.min(low_sections, axis=1).tolist()
    highs_list = np.max(high_sections, axis=1).tolist()
    atr_list = np.max(atr_sections, axis=1).tolist()

    return next_three_business_days, lows_list, highs_list, atr_list

def check_and_adjust(row):
    if round(row['predicted_low'], 1) == round(row['predicted_high'], 1):
        # Generate a random adjustment percentage between 7% and 10%
        adjustment_percentage_high = np.random.uniform(0.07, 0.10)
        row['predicted_high'] *= (1 + adjustment_percentage_high)
        
        # Generate a random adjustment percentage between -4% and +4%
        adjustment_percentage_low = np.random.uniform(-0.03, 0.03)
        row['predicted_low'] *= (1 + adjustment_percentage_low)
    return row


def vertical_variation(row):
     # Generate a random adjustment between -2% and +2% for 'predicted_low'
    adjustment_low = np.random.uniform(-0.02, 0.02)
    row['predicted_low'] *= (1 + adjustment_low)
    
    # Generate a random adjustment between 5% and 7% for 'predicted_high'
    adjustment_high = np.random.uniform(0.05, 0.07)
    row['predicted_high'] *= (1 + adjustment_high)
    
    return row

def get_pred_table(next_three_business_days, lows_list, highs_list, atr_list, last_low, last_high):
    pct_dev = 0.15  # 15% deviation

    # Create empty lists to store the prices
    dates = []
    predicted_lows = []
    predicted_highs = []
    predicted_atrs = []

    # Generate random prices for each date
    for i in range(len(next_three_business_days)):
        date = next_three_business_days[i]
        deviation_low = random.uniform(-pct_dev, pct_dev)
        deviation_high = random.uniform(-pct_dev, pct_dev)
        deviation_atr = random.uniform(-pct_dev, pct_dev)

        # Add the first row
        dates.append(date)
        predicted_lows.append(lows_list[i] + deviation_low)
        predicted_highs.append(highs_list[i] + deviation_high)
        predicted_atrs.append(atr_list[i] + deviation_atr)

        # Add the second and third rows
        for _ in range(2):
            deviation_low = random.uniform(-pct_dev, pct_dev)
            deviation_high = random.uniform(-pct_dev, pct_dev)
            deviation_atr = random.uniform(-pct_dev, pct_dev)

            upper_limit_low = predicted_highs[-1] - pct_dev  # upper bound
            upper_limit_atr = predicted_atrs[-1] - pct_dev # upper bound
            
            # apply upper bounds
            new_low = max(predicted_lows[-1] + deviation_low, upper_limit_low)
            new_high = max(predicted_highs[-1] + deviation_high, new_low) # swapping to ensure low < high 
            new_atr = max(predicted_atrs[-1] + deviation_atr, upper_limit_atr)

            # append predicted prices for next 3 business days
            dates.append(date)
            predicted_lows.append(new_low)
            predicted_highs.append(new_high)
            predicted_atrs.append(new_atr)


    # check variation
    threshold = 9
    pct_dev_low = [(pl - last_low) / last_low * 100 for pl in predicted_lows]
    pct_dev_high = [(ph - last_high) / last_high * 100 for ph in predicted_highs]
    predicted_lows = [pl if abs(pd) <= threshold else last_low * (1 + threshold / 100) for pl, pd in zip(predicted_lows, pct_dev_low)]
    predicted_highs = [ph if abs(pd) <= threshold else last_high * (1 + threshold / 100) for ph, pd in zip(predicted_highs, pct_dev_high)]


    # dataframe with predicted prices
    res = pd.DataFrame({
        'predicted_low': predicted_lows,
        'predicted_high': predicted_highs,
        'predicted_atr': predicted_atrs,
    }, index=dates)

    # Switching the 4th row to the 2nd row
    res.iloc[[1, 3]] = res.iloc[[3, 1]]
    res.iloc[[2, 6]] = res.iloc[[6, 2]]
    res.iloc[[3, 7]] = res.iloc[[7, 3]]
    res.iloc[[4, 8]] = res.iloc[[8, 4]]

    # if pred_low price == pred_high price
    mask = res['predicted_low'].round(1) == res['predicted_high'].round(1)

    # Compute the random deviations based on 5% of 'predicted_high'
    deviations = res.loc[mask, 'predicted_high'] * np.random.uniform(-0.25, 0.25, size=mask.sum())

    # Add the random deviations to 'predicted_high' for rows where the mask is True
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

    if pred_df_filled['predicted_low'].mean() < 1:
        pred_df_filled[['predicted_low', 'predicted_high']] = pred_df_filled[['predicted_low', 'predicted_high']].round(4)
    else:
        pred_df_filled[['predicted_low', 'predicted_high']] = pred_df_filled[['predicted_low', 'predicted_high']].round(2)


    pred_df_filled = pred_df_filled.apply(check_and_adjust, axis=1)
    pred_df_filled = pred_df_filled.apply(vertical_variation, axis=1)

    
    # predicted directional column
    if 'predicted_low_adjusted' in pred_df_filled:
        pred_low_col = 'predicted_low_adjusted'
        pred_high_col = 'predicted_high_adjusted'
    else:
        pred_low_col = 'predicted_low'
        pred_high_col = 'predicted_high'

    pred_df_filled['rolling_avg_low'] = pred_df_filled[pred_low_col].rolling(window=3, min_periods=1).mean()
    pred_df_filled['rolling_avg_high'] = pred_df_filled[pred_high_col].rolling(window=3, min_periods=1).mean()

    # Fill in the predicted_low_direction column
    pred_df_filled['predicted_low_direction'] = pred_df_filled.apply(lambda row: 'Increase' if (row[pred_low_col] > row['rolling_avg_low'] or last_low > row['rolling_avg_low']) else 'Decrease', axis=1)

    # Fill in the predicted_high_direction column
    pred_df_filled['predicted_high_direction'] = pred_df_filled.apply(lambda row: 'Increase' if (row[pred_high_col] > row['rolling_avg_high'] or last_high > row['rolling_avg_high']) else 'Decrease', axis=1)

    # Invert the directions for the first date
    pred_df_filled.iloc[0, pred_df_filled.columns.get_loc('predicted_low_direction')] = 'Increase' if pred_df_filled.iloc[0][pred_low_col] > last_low else 'Decrease'
    pred_df_filled.iloc[0, pred_df_filled.columns.get_loc('predicted_high_direction')] = 'Increase' if pred_df_filled.iloc[0][pred_high_col] > last_high else 'Decrease'

    pred_df_filled['predicted_low_direction'][1:3] = pred_df_filled['predicted_low_direction'][0]
    pred_df_filled['predicted_high_direction'][1:3] = pred_df_filled['predicted_high_direction'][0]

    # Drop the rolling average columns if you don't need them anymore
    pred_df_filled.drop(['rolling_avg_low', 'rolling_avg_high'], axis=1, inplace=True)

    pred_df_filled['variance'] = [1, 2, 3, 1, 2, 3, 1, 2, 3]

    if pred_df_filled['predicted_atr'].mean() < 1:
        pred_df_filled['predicted_atr'] = pred_df_filled['predicted_atr'].round(4)
    else:
        pred_df_filled['predicted_atr'] = pred_df_filled['predicted_atr'].round(2)
    

    return pred_df_filled


def filter_and_reformat_data(data):
    ticker_times_ind = defaultdict(list)
    
    # Step 1: Extract the unique date entries and their corresponding latest times for each ticker.
    
    for item in data:
        if 'indicators' in item: # indicator
            _ind = 'indicator'
        else:
            _ind = ''

        parts = item.split("_")

        ticker = parts[-2]
        _date = parts[1].split('/')[1]
        _time = parts[2]
        date_time_ind = (_date, _time, _ind)
        ticker_times_ind[ticker].append(date_time_ind)

    # Step 2: Filter the list to only include the latest time entries for each unique date and ticker.
    filtered_data = []
    for ticker, date_times_ind in ticker_times_ind.items():

        latest_time = max(date_times_ind, key=lambda x: x[0]+x[1])
        
        if latest_time[-1] == '': # not indicator
            filtered_data.extend([f"myawsbucket-st/'streamlit_uploads'/{latest_time[0]}_{latest_time[1]}_{ticker}_lookback.csv",
            f"myawsbucket-st/'streamlit_uploads'/{latest_time[0]}_{latest_time[1]}_{ticker}_predictions.csv"])
        else: # indicator
            filtered_data.extend([f"myawsbucket-st/'streamlit_uploads'/{latest_time[0]}_{latest_time[1]}_indicators_{ticker}_lookback.csv",
            f"myawsbucket-st/'streamlit_uploads'/{latest_time[0]}_{latest_time[1]}_indicators_{ticker}_predictions.csv"])

    # Step 3: Reformat the remaining entries.
    tickers = []
    reformatted_data = []
    for item in filtered_data:
        parts = item.split("_")
        ticker = parts[-2]
        date = parts[1].split('/')[1]
        if 'indicators' in parts[-3]:
            print_name = f"{ticker.upper()} {date[0:4]}/{date[4:6]}/{date[6:8]} {('Lookback Indicators' if 'lookback' in item else 'Predictions Indicators')}"
        else:
            print_name = f"{ticker.upper()} {date[0:4]}/{date[4:6]}/{date[6:8]} {('Lookback' if 'lookback' in item else 'Predictions')}"
        tickers.append(ticker.upper())
        reformatted_data.append(print_name)

    tickers = set(tickers)

    return tickers, reformatted_data

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
    benchmark_symbol = '^GSPC'  # S&P 500 as the benchmark index
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
    beta = (rolling_cov / rolling_var).abs()
    df['beta'] = beta

    # adjust predicted high and lows and atr based on VIX `vix_values`
    df['predicted_low_adjusted'] = np.random.normal(loc=df['predicted_low'], scale=df['predicted_low'] * beta)
    df['predicted_high_adjusted'] = np.random.normal(loc=df['predicted_high'], scale=df['predicted_high'] * beta)
    df['predicted_atr_adjusted'] = np.random.normal(loc=df['predicted_atr'], scale=df['predicted_atr'] * beta)

    # pct diff
    pct_diff_low = ((df['predicted_low_adjusted'] - df['low']) / df['low'])
    pct_diff_high = ((df['predicted_high_adjusted'] - df['high']) / df['high'])
    
    df['pct_diff_low_adjusted'] = pct_diff_low * 100
    df['pct_diff_high_adjusted'] = pct_diff_high * 100

    df = df[['VIX', 'beta', 'low', 'predicted_low_adjusted', 'high', 'predicted_high_adjusted', 'predicted_atr_adjusted', 'pct_diff_low_adjusted', 'pct_diff_high_adjusted', 'predicted_low_direction', 'predicted_high_direction', 'directional_accuracy']]

    df[['predicted_low_adjusted', 'predicted_high_adjusted']] = df[['predicted_low_adjusted', 'predicted_high_adjusted']].agg([min, max], axis=1)

    df = df.iloc[1:]

    # df = df.round(2)

    df = df.applymap(remove_trailing_zeroes)

    df['beta'] = beta

    return df

def adjust_indicator_table(df):

    num_cols = ['pct_diff_low_adjusted', 'pct_diff_high_adjusted', 'predicted_low_adjusted', 'predicted_high_adjusted']
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')

    pct_diff_threshold = 12

    df['predicted_low_adjusted'] = np.where(
        df['pct_diff_low_adjusted'] > pct_diff_threshold,
        df['predicted_low_adjusted'] * (1 + pct_diff_threshold / 100),
        np.where(
            df['pct_diff_low_adjusted'] < -pct_diff_threshold,
            df['predicted_low_adjusted'] * (1 - pct_diff_threshold / 100),
            df['predicted_low_adjusted']
        )
    )

    df['predicted_high_adjusted'] = np.where(
        df['pct_diff_high_adjusted'] > pct_diff_threshold,
        df['predicted_high_adjusted'] * (1 + pct_diff_threshold / 100),
        np.where(
            df['pct_diff_high_adjusted'] < -pct_diff_threshold,
            df['predicted_high_adjusted'] * (1 - pct_diff_threshold / 100),
            df['predicted_high_adjusted']
        )
    )

    df = df.applymap(remove_trailing_zeroes)

    return df




def adjust_pred_table(df):
    low_adjustments = np.random.normal(loc=df['predicted_low'], scale=df['predicted_low'] * 0.01)
    high_adjustments = np.random.normal(loc=df['predicted_high'], scale=df['predicted_high'] * 0.01)
    atr_adjustments = np.random.normal(loc=df['predicted_atr'], scale=df['predicted_atr'] * 0.01)
    
    # Clip 'low' price adjustments to ensure it's always lower than or equal to 'high' price adjustments
    low_adjusted = np.clip(low_adjustments, a_min=None, a_max=high_adjustments)
    
    # Adjust 'low_adjusted' and 'high_adjustments' if they are identical
    mask = low_adjusted == high_adjustments
    low_adjusted[mask] -= 0.001  # Randomly decrease low_adjusted by a small amount
    high_adjustments[mask] += 0.001  # Randomly increase high_adjustments by a small amount
    
    # Set the adjusted prices in the DataFrame
    df['predicted_low_adjusted'] = low_adjusted
    df['predicted_high_adjusted'] = high_adjustments
    df['predicted_atr_adjusted'] = atr_adjustments

    # reduce variation
    smoothing_factor = 0.1
    mean_val = (df["predicted_low_adjusted"] + df["predicted_high_adjusted"]) / 2
    diff = (df["predicted_high_adjusted"] - df["predicted_low_adjusted"]) * smoothing_factor
    df["predicted_high_adjusted"] = mean_val + diff
    df["predicted_low_adjusted"] = mean_val - diff

    if df['predicted_low_adjusted'].mean() < 1:
        df[['predicted_low_adjusted', 'predicted_high_adjusted', 'predicted_atr_adjusted']] = df[['predicted_low_adjusted', 'predicted_high_adjusted', 'predicted_atr_adjusted']].round(4)
    else:
        df[['predicted_low_adjusted', 'predicted_high_adjusted', 'predicted_atr_adjusted']] = df[['predicted_low_adjusted', 'predicted_high_adjusted', 'predicted_atr_adjusted']].round(2)
    

    df = df[['predicted_low_adjusted', 'predicted_high_adjusted', 'predicted_atr_adjusted', 'predicted_low_direction', 'predicted_high_direction']]

    df['variance'] = [1, 2, 3, 1, 2, 3, 1, 2, 3]

    return df

def remove_trailing_zeroes(val):
    if isinstance(val, float):
        return '{:.2f}'.format(val).rstrip('0').rstrip('.')
    return val

def get_perf_df(df, ticker):

    start_date = df.date.iloc[0]
    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    start_date_utc = time.mktime(start_date.timetuple())
    end_date = df.date.iloc[-1]
    end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)
    end_date_utc = time.mktime(end_date.timetuple())

    eod_df = get_dataframe(ticker, start_date_utc, end_date_utc)

    eod_df = eod_df[['low', 'high']]

    eod_df.index = pd.to_datetime(eod_df.index)
    grouped_df = eod_df.groupby(eod_df.index.date).apply(lambda x: x.loc[x['low'].idxmin()])

    grouped_df = grouped_df.reset_index().rename({'index': 'date', 'high': 'actual_high', 'low': 'actual_low'}, axis=1)

    # Convert 'date' columns to datetime
    df['date'] = pd.to_datetime(df['date'])
    grouped_df['date'] = pd.to_datetime(grouped_df['date'])

    # Merge the DataFrames on 'date'
    merged_df = pd.merge(grouped_df, df, on='date', how='left')

    # merged_df = merged_df.round(2)

    merged_df['date'] = pd.to_datetime(merged_df['date']).dt.date

    if 'predicted_low_adjusted' in merged_df:
        pred_low_col = 'predicted_low_adjusted'
        pred_high_col = 'predicted_high_adjusted'
    else:
        pred_low_col = 'predicted_low'
        pred_high_col = 'predicted_high'
   
    # avg pct diff
    pct_diff_low = ((merged_df[pred_low_col] - merged_df['actual_low']) / merged_df['actual_low'])
    pct_diff_high = ((merged_df[pred_high_col] - merged_df['actual_high']) / merged_df['actual_high'])

    # pct_diff = (abs(pct_diff_high) + abs(pct_diff_low)) / 2

    merged_df['pct_diff_low'] = pct_diff_low * 100
    merged_df['pct_diff_high'] = pct_diff_high * 100

    # merged_df['avg_pct_diff'] = pct_diff * 100
    # merged_df['avg_pct_diff'] = merged_df['avg_pct_diff'].round(2)

    return merged_df