from pstats import Stats
import numpy as np
import urllib.request
import json
import random
import time
import datetime
from datetime import date, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.impute import KNNImputer
from collections import defaultdict
import os
os.environ['PYTHONHASHSEED']=str(1)
import tensorflow as tf
import yfinance as yf
import ta
pd.set_option('mode.chained_assignment', None)
pd.set_option('display.float_format', '{:.2f}'.format)

# set seeds
os.environ['PYTHONHASHSEED']=str(1)
tf.random.set_seed(1)
np.random.seed(1)
random.seed(1)
EHD_API_KEY = st.secrets["EHD_API_KEY"]

@st.cache_data(ttl=24*3600, max_entries=3)
def get_dataframe_eod(ticker, start_date_utc, end_date_utc):
    url = 'https://eodhistoricaldata.com/api/intraday/{}?api_token={}&order=d&interval=1h&fmt=json&from={}&to={}'.format(ticker, EHD_API_KEY, start_date_utc, end_date_utc)
    response = urllib.request.urlopen(url)
    eod_data = json.loads(response.read())
    eod_data_df = pd.json_normalize(eod_data)
    eod_data_df['datetime'] = pd.to_datetime(eod_data_df['datetime'])
    eod_data_df.set_index('datetime', inplace=True)
    return eod_data_df

@st.cache_data(ttl=24*3600, max_entries=3)
def get_dataframe_yf(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)

    # lowercase
    df.index.names = ['datetime']
    df.columns = [x.lower() for x in df.columns]

    return df

def load_chart(low_high_df, ticker):
    sma_df = low_high_df.copy()
    st.subheader('SMA Chart - ${}'.format(ticker.upper()))
    sma_df['sma_5'] = sma_df['close'].rolling(window=5).mean()
    sma_df['sma_20'] = sma_df['close'].rolling(window=20).mean()
    return st.line_chart(sma_df[['close', 'sma_5', 'sma_20']])

def prepare_data(data, time_steps=1):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps), 0])
        y.append(data[i + time_steps, 0])
    return np.array(X), np.array(y)

def crypto_date_filter(low_high_df):

    business_days = low_high_df[low_high_df.index.weekday < 5]  # select rows for Monday (0) to Friday (4)
    selected_hours = business_days.between_time('00:00:00', '20:00:00', include_end=False).iloc[::2]

    filtered_df = low_high_df.loc[selected_hours.index]

    return filtered_df

@st.cache_data(ttl=24*3600, max_entries=3)
def get_atr_dataframe(low_high_df):
    # get atr first
    low_high_df['true_range'] = low_high_df['high'] - low_high_df['low']
    low_high_df['high_shifted'] = low_high_df['high'].shift(1)
    low_high_df['low_shifted'] = low_high_df['low'].shift(1)
    low_high_df['high_minus_close_shifted'] = abs(low_high_df['high'] - low_high_df['close'].shift(1))
    low_high_df['low_minus_close_shifted'] = abs(low_high_df['low'] - low_high_df['close'].shift(1))

    low_high_df['true_range'] = low_high_df[['true_range', 'high_shifted', 'low_shifted', 'high_minus_close_shifted', 'low_minus_close_shifted']].apply(lambda row: max(row['true_range'], row['high_shifted'] - row['low_shifted'], row['high_minus_close_shifted'], row['low_minus_close_shifted']), axis=1)

    low_high_df.drop(columns=['high_shifted', 'low_shifted', 'high_minus_close_shifted', 'low_minus_close_shifted'], inplace=True)

    # calculate Average True Range (ATR)
    low_high_df['atr'] = low_high_df['true_range'].rolling(window=1).mean()

    return low_high_df

def get_correlation_matrix(df):
    df = df.apply(pd.to_numeric, errors='coerce')
    correlation_matrix = df.corr()
    corr = correlation_matrix[['low', 'high']]
    corr = corr.drop(['open', 'low', 'high', 'close', 'volume'], axis=0)
    # only drop 'date' if exists
    if 'date' in corr.index:
        corr = corr.drop(['date'], axis=0)
    st.write(corr)
    return None

@st.cache_data(ttl=24*3600, max_entries=3)
def initiate_model(ticker, low_high_df):
    # model
    low_prices = low_high_df['low'].values.reshape(-1, 1)
    high_prices = low_high_df['high'].values.reshape(-1, 1)
    
    # scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data_low = scaler.fit_transform(low_prices)
    scaled_data_high = scaler.fit_transform(high_prices)

    # train test split
    train_data_low, test_data_low, train_data_high, test_data_high = train_test_split(scaled_data_low, scaled_data_high, test_size=0.2, shuffle=False, random_state=1)

    # prepare the data for LSTM
    time_steps = 1  # number of previous time steps to use for prediction
    x_train_low, y_train_low = prepare_data(train_data_low, time_steps)
    x_test_low, y_test_low = prepare_data(test_data_low, time_steps)

    x_train_high, y_train_high = prepare_data(train_data_high, time_steps)
    x_test_high, y_test_high = prepare_data(test_data_high, time_steps)

    # reshape the input data to fit the LSTM model input shape
    x_train_low = np.reshape(x_train_low, (x_train_low.shape[0], x_train_low.shape[1], 1))
    x_test_low = np.reshape(x_test_low, (x_test_low.shape[0], x_test_low.shape[1], 1))

    x_train_high = np.reshape(x_train_high, (x_train_high.shape[0], x_train_high.shape[1], 1))
    x_test_high = np.reshape(x_test_high, (x_test_high.shape[0], x_test_high.shape[1], 1))

    # LSTM model
    model = Sequential(name='lstm_model')
    model.add(LSTM(128, return_sequences=True, input_shape=(x_train_low.shape[1], 1), name='lstm_1'))
    model.add(LSTM(64, return_sequences=False, name='lstm_2'))
    model.add(Dense(25, name='dense_1'))
    model.add(Dense(1, name='dense_2'))
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model, scaled_data_low, scaled_data_high, x_train_low, y_train_low, x_test_low, x_train_high, y_train_high, x_test_high

@st.cache_resource(ttl=24*3600, max_entries=3)
def run_model(_model, low_high_df, train_size, time_steps, scaled_data, x_test, x_train, y_train, col_name):

    # fit
    low_prices = low_high_df['low'].values.reshape(-1, 1)
    high_prices = low_high_df['high'].values.reshape(-1, 1)
    _model.fit(x_train, y_train, batch_size=1, epochs=15, verbose=0)
    scaler = MinMaxScaler(feature_range=(0, 1))

    if col_name=='predictions_low':
        scaled_data = scaler.fit_transform(low_prices)
        last_price = low_high_df['low'].iloc[-1]
        
    elif col_name=='predictions_high':
        scaled_data = scaler.fit_transform(high_prices)
        last_price = low_high_df['high'].iloc[-1]
        
    # predict
    predictions = _model.predict(x_test, verbose=0)
    predictions = scaler.inverse_transform(predictions)

    # get last price
    tolerance_percentage = 3
    current_price = predictions[-1][0]
    threshold = current_price * (1 - tolerance_percentage / 100.0)
    
    # assign predictions
    valid = low_high_df[train_size:-1]
    valid[col_name] = predictions

    # prepare the next day's input data
    next_day_input = np.array([scaled_data[-time_steps:, 0]])
    next_day_input = np.reshape(next_day_input, (next_day_input.shape[0], next_day_input.shape[1], 1))

    # predictions for the next 21 values
    predictions_list = []
    for _ in range(21):
        next_day_prediction = _model.predict(next_day_input, verbose=0)
        next_day_prediction = np.minimum(next_day_prediction, threshold)
        predictions_list.append(next_day_prediction)
        next_day_input = np.append(next_day_input[:, 1:, :], np.expand_dims(next_day_prediction, axis=1), axis=1)

    # reshape predictions to match expected input of inverse_transform
    predictions_list = np.reshape(predictions_list, (len(predictions_list), 1))
    predicted = scaler.inverse_transform(predictions_list)
    
    lower_threshold = float(last_price) * (1 - float(tolerance_percentage) / 100.0)
    upper_threshold = float(last_price) * (1 + float(tolerance_percentage) / 100.0)

    # clip predictions to be within the tolerance range
    predicted_clipped = np.where(np.logical_or(predicted < lower_threshold, predicted > upper_threshold),
                                    np.random.uniform(lower_threshold, upper_threshold, size=21),
                                    predicted)

    erratic_noise = np.random.uniform(-1, 1, size=21)
    predicted_clipped = predicted_clipped + erratic_noise
    predicted = predicted_clipped[-1]

    return predictions, predicted

@st.cache_data(ttl=24*3600, max_entries=3)
def get_grouped_df(df): # turn predictions into table
    df.index = pd.to_datetime(df.index)

    # group and select minimum lows
    df['low'] = pd.to_numeric(df['low'], errors='coerce')
    df['high'] = pd.to_numeric(df['high'], errors='coerce')
    df['date'] = df.index.date
    grouped_df = df.groupby(df.index.date).apply(lambda x: x.iloc[-1])
    # grouped_df = df.groupby(df.index.date)


    # swapping predicted low and high to ensure low is always the lowest
    grouped_df.loc[grouped_df['predictions_low'] > grouped_df['predictions_high'], ['predictions_low', 'predictions_high']] = grouped_df.loc[grouped_df['predictions_low'] > grouped_df['predictions_high'], ['predictions_high', 'predictions_low']].values

    # pct diff
    pct_diff_low = ((grouped_df['predictions_low'] - grouped_df['low']) / grouped_df['low'])
    pct_diff_high = ((grouped_df['predictions_high'] - grouped_df['high']) / grouped_df['high'])

    grouped_df['pct_diff_low'] = pct_diff_low * 100
    grouped_df['pct_diff_high'] = pct_diff_high * 100

    # direction prediction
    pred_low_col, pred_high_col = 'predictions_low', 'predictions_high'

    # predicted_low_direction
    grouped_df['diff_low'] = grouped_df[pred_low_col] - grouped_df[pred_low_col].shift(1)
    grouped_df['predicted_low_direction'] = grouped_df['diff_low'].apply(lambda x: 'Increase' if x > 0 else 'Decrease')
    grouped_df.drop(columns=['diff_low'], inplace=True)

    # predicted_high_direction
    grouped_df['diff_high'] = grouped_df[pred_high_col] - grouped_df[pred_high_col].shift(1)
    grouped_df['predicted_high_direction'] = grouped_df['diff_high'].apply(lambda x: 'Increase' if x > 0 else 'Decrease')
    grouped_df.drop(columns=['diff_high'], inplace=True)

    # move columns
    grouped_df.insert(2, "predictions_low", grouped_df.pop('predictions_low'))
    grouped_df.insert(4, "predictions_high", grouped_df.pop('predictions_high'))
    
    # rename
    grouped_df = grouped_df.rename(columns={'predictions_low': 'predicted_low', 'predictions_high': 'predicted_high'})

    return grouped_df

def is_business_day(date_obj):
    return date_obj.isoweekday() <= 5

@st.cache_data(ttl=24*3600, max_entries=3)
def predict(end_date, predicted_low, predicted_high):
    # get the next three business days
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

@st.cache_data(ttl=24*3600, max_entries=3)

def get_pred_table(next_three_business_days, lows_list, highs_list, last_low, last_high, last_close):
    pct_dev = np.random.uniform(-0.15, 0.15)  # 15% deviation
    lows_list = [last_low + (low - last_low) * 0.15 for low in lows_list]
    highs_list = [last_high + (high - last_high) * 0.15 for high in highs_list]

    # Create empty lists to store the prices
    dates = []
    predicted_lows = []
    predicted_highs = []

    # Generate random prices for each date
    for i in range(len(next_three_business_days)):
        date = next_three_business_days[i]
        deviation_low = random.uniform(-pct_dev, pct_dev)
        deviation_high = random.uniform(-pct_dev, pct_dev)

        # Add the first row
        dates.append(date)
        predicted_lows.append(lows_list[i] + deviation_low)
        predicted_highs.append(highs_list[i] + deviation_high)

        # Add the second and third rows
        for _ in range(2):
            deviation_low = random.uniform(-pct_dev, pct_dev)
            deviation_high = random.uniform(-pct_dev, pct_dev)
            upper_limit_low = predicted_highs[-1] - pct_dev  # upper bound
            
            # apply upper bounds
            new_low = max(predicted_lows[-1] + deviation_low, upper_limit_low)
            new_high = max(predicted_highs[-1] + deviation_high, new_low) 

            # append predicted prices for next 3 business days
            dates.append(date)
            predicted_lows.append(new_low)
            predicted_highs.append(new_high)

    # check variation
    threshold = 6
    pct_dev_low = [(pl - last_low) / last_low * 100 for pl in predicted_lows]
    pct_dev_high = [(ph - last_high) / last_high * 100 for ph in predicted_highs]
    predicted_lows = [pl if abs(pd) <= threshold else last_low * (1 + threshold / 100) for pl, pd in zip(predicted_lows, pct_dev_low)]
    predicted_highs = [ph if abs(pd) <= threshold else last_high * (1 + threshold / 100) for ph, pd in zip(predicted_highs, pct_dev_high)]

    # dataframe with predicted prices
    res = pd.DataFrame({
        'predicted_low': predicted_lows,
        'predicted_high': predicted_highs,
    }, index=dates)

    # Switching the 4th row to the 2nd row
    res.iloc[[1, 3]] = res.iloc[[3, 1]]
    res.iloc[[2, 6]] = res.iloc[[6, 2]]
    res.iloc[[3, 7]] = res.iloc[[7, 3]]
    res.iloc[[4, 8]] = res.iloc[[8, 4]]

    # if pred_low price == pred_high price
    mask = res['predicted_low'].round(1) == res['predicted_high'].round(1)

    # Compute the random deviations based on 5% of 'predicted_high'
    deviations = res.loc[mask, 'predicted_high'] * np.random.uniform(-0.25, 0.125, size=mask.sum())

    # Add the random deviations to 'predicted_high' for rows where the mask is True
    res.loc[mask, 'predicted_high'] += deviations

    # check NaNs
    pred_df_filled = res.copy()
    imputer = KNNImputer(n_neighbors=5)
    pred_df_filled['predicted_high'] = imputer.fit_transform(pred_df_filled[['predicted_high']])
    pred_df_filled['predicted_high'] = pred_df_filled[['predicted_low', 'predicted_high']].max(axis=1)

    # reduce variation
    smoothing_factor = 0.15
    mean_val = (pred_df_filled["predicted_low"] + pred_df_filled["predicted_high"]) / 2
    diff = (pred_df_filled["predicted_high"] - pred_df_filled["predicted_low"]) * smoothing_factor
    pred_df_filled["predicted_high"] = mean_val + diff
    pred_df_filled["predicted_low"] = mean_val - diff

    # adjust rounding based on mean
    if pred_df_filled['predicted_low'].mean() < 1:
        pred_df_filled[['predicted_low', 'predicted_high']] = pred_df_filled[['predicted_low', 'predicted_high']].round(4)
    else:
        pred_df_filled[['predicted_low', 'predicted_high']] = pred_df_filled[['predicted_low', 'predicted_high']].round(2)

    # predicted directional column
    if 'predicted_low_adjusted' in pred_df_filled:
        pred_low_col = 'predicted_low_adjusted'
        pred_high_col = 'predicted_high_adjusted'
    else:
        pred_low_col = 'predicted_low'
        pred_high_col = 'predicted_high'

    # prepare rolling average columns for direction columns
    pred_df_filled['rolling_avg_low'] = pred_df_filled[pred_low_col].rolling(window=3, min_periods=1).mean()
    pred_df_filled['rolling_avg_high'] = pred_df_filled[pred_high_col].rolling(window=3, min_periods=1).mean()

    # fill in the predicted_low_direction column
    pred_df_filled['predicted_low_direction'] = pred_df_filled.apply(lambda row: 'Increase' if (row[pred_low_col] > row['rolling_avg_low'] or last_low > row['rolling_avg_low']) else 'Decrease', axis=1)

    # fill in the predicted_high_direction column
    pred_df_filled['predicted_high_direction'] = pred_df_filled.apply(lambda row: 'Increase' if (row[pred_high_col] > row['rolling_avg_high'] or last_high > row['rolling_avg_high']) else 'Decrease', axis=1)

    # invert the directions for the first date
    pred_df_filled.iloc[0, pred_df_filled.columns.get_loc('predicted_low_direction')] = 'Increase' if pred_df_filled.iloc[0][pred_low_col] > last_low else 'Decrease'
    pred_df_filled.iloc[0, pred_df_filled.columns.get_loc('predicted_high_direction')] = 'Increase' if pred_df_filled.iloc[0][pred_high_col] > last_high else 'Decrease'

    # fill in the predicted_low_direction and predicted_high_direction columns for the second and third dates
    pred_df_filled['predicted_low_direction'][1:3] = pred_df_filled['predicted_low_direction'][0]
    pred_df_filled['predicted_high_direction'][1:3] = pred_df_filled['predicted_high_direction'][0]

    pred_df_filled.drop(['rolling_avg_low', 'rolling_avg_high'], axis=1, inplace=True)
    pred_df_filled['variance'] = [1, 2, 3, 1, 2, 3, 1, 2, 3]

    # check if predicted prices are the same as the previous day's prices
    condition_low = pred_df_filled['predicted_low'].eq(pred_df_filled['predicted_low'].shift(1))
    condition_high = pred_df_filled['predicted_high'].eq(pred_df_filled['predicted_high'].shift(1))

    # update 'predicted_low' and 'predicted_high' based on the conditions
    pred_df_filled['predicted_low'] = np.where(condition_low, pred_df_filled.apply(lambda row: generate_random_value(row['predicted_low']), axis=1), pred_df_filled['predicted_low'])
    pred_df_filled['predicted_high'] = np.where(condition_high, pred_df_filled.apply(lambda row: generate_random_value(row['predicted_high']), axis=1), pred_df_filled['predicted_high'])

    # check if 'predicted_low' is the same as 'predicted_high' in the current row
    mask = pred_df_filled['predicted_low'].round(1) == pred_df_filled['predicted_high'].round(1)
    deviations = pred_df_filled.loc[mask, 'predicted_high'] * np.random.uniform(0.05, 0.125, size=mask.sum())
    pred_df_filled.loc[mask, 'predicted_high'] += deviations
    pred_df_filled.loc[mask, 'predicted_low'] -= deviations

    # damping deviation
    pred_df_filled['predicted_low'] = last_low + (pred_df_filled['predicted_low'] - last_low) * 0.15
    pred_df_filled['predicted_high'] = last_high + (pred_df_filled['predicted_high'] - last_high) * 0.15

    # also round low and high
    if pred_df_filled['predicted_low'].mean() < 1:
        pred_df_filled['predicted_low'] = pred_df_filled['predicted_low'].round(4)
    else:
        pred_df_filled['predicted_low'] = pred_df_filled['predicted_low'].round(2)

    if pred_df_filled['predicted_high'].mean() < 1:
        pred_df_filled['predicted_high'] = pred_df_filled['predicted_high'].round(4)
    else:
        pred_df_filled['predicted_high'] = pred_df_filled['predicted_high'].round(2)
    
    # predicted_atr
    pred_df_filled['predicted_atr'] = pred_df_filled.apply(calculate_atr, axis=1, last_close=last_close, pred_low_col=pred_low_col, pred_high_col=pred_high_col)

    # move columns
    pred_df_filled.insert(2, "predicted_atr", pred_df_filled.pop('predicted_atr'))

    # make sure predicted_low < predicted_high
    pred_df_filled.loc[pred_df_filled['predicted_low'] > pred_df_filled['predicted_high'], ['predicted_low', 'predicted_high']] = pred_df_filled.loc[pred_df_filled['predicted_low'] > pred_df_filled['predicted_high'], ['predicted_high', 'predicted_low']].values

    return pred_df_filled

def generate_random_value(value, deviation=0.1):
        return np.random.uniform(value - deviation, value + deviation)

def calculate_atr(row, pred_low_col, pred_high_col, last_close):
    predicted_low = row[pred_low_col]
    predicted_high = row[pred_high_col]
    
    # calculate the True Range (TR) for the row
    tr = max(predicted_high - predicted_low, abs(predicted_high - last_close), abs(predicted_low - last_close))
    
    return tr

def filter_and_reformat_data(data):
    ticker_times_ind = defaultdict(list)
    
    # group list by ticker
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

    # filter out the latest entries
    filtered_data = []
    for ticker, date_times_ind in ticker_times_ind.items():

        latest_time = max(date_times_ind, key=lambda x: x[0]+x[1])
        
        if latest_time[-1] == '': # not indicator
            filtered_data.extend([f"myawsbucket-st/'streamlit_uploads'/{latest_time[0]}_{latest_time[1]}_{ticker}_lookback.csv",
            f"myawsbucket-st/'streamlit_uploads'/{latest_time[0]}_{latest_time[1]}_{ticker}_predictions.csv"])
        else: # indicator
            filtered_data.extend([f"myawsbucket-st/'streamlit_uploads'/{latest_time[0]}_{latest_time[1]}_indicators_{ticker}_lookback.csv",
            f"myawsbucket-st/'streamlit_uploads'/{latest_time[0]}_{latest_time[1]}_indicators_{ticker}_predictions.csv"])

    # reformat data
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
    # find files with substrings in AWS
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

@st.cache_data(ttl=24*3600, max_entries=3)
def append_indicators(df, start_date, end_date):

    # convert dates to strings
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    # extract yf data
    vix_data = yf.download('^VIX', start=start_date_str, end=end_date_str)
    sp500_data = yf.download('^GSPC', start=start_date_str, end=end_date_str)

    # extend vix_data to df by date
    vix_data = vix_data.reindex(df.index, method='ffill')
    sp500_data = sp500_data.reindex(df.index, method='ffill')
    vix_values = vix_data['Close']
    sp500_values = sp500_data['Close']
    df['VIX'] = vix_values
    df['sp500'] = sp500_values

    # beta
    rolling_cov = df['close'].rolling(window=2).cov(df['sp500'])
    rolling_var = df['sp500'].rolling(window=2).var()
    beta = (rolling_cov / rolling_var).abs()
    df['beta'] = beta

    # rsi
    df['rsi_14'] = ta.momentum.RSIIndicator(df['close'], window=14, fillna=True).rsi()
    df['rsi_21'] = ta.momentum.RSIIndicator(df['close'], window=21, fillna=True).rsi()

    # macd
    df['macd'] = ta.trend.macd(df['close'], fillna=True)

    # bollinger bands
    indicator_bb = ta.volatility.BollingerBands(close=df["close"], window=20, window_dev=2, fillna=True)
    df['bb_mavg'] = indicator_bb.bollinger_mavg()

    # clean up
    df = df.iloc[1:]
    df = df.applymap(remove_trailing_zeroes)

    return df

def adjust_indicator_table(df):

    num_cols = ['pct_diff_low_adjusted', 'pct_diff_high_adjusted', 'predicted_low_adjusted', 'predicted_high_adjusted']
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')

    pct_diff_threshold = 12

    # adjust predicted prices based on pct_diff_threshold
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
    
    # clip the adjustments to be within the tolerance range
    low_adjusted = np.clip(low_adjustments, a_min=None, a_max=high_adjustments)
    
    # adjust 'low_adjusted' and 'high_adjustments' if they are identical
    mask = low_adjusted == high_adjustments
    low_adjusted[mask] -= 0.001 
    high_adjustments[mask] += 0.001 
    
    # set adjusted prices in the DataFrame
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

    # convert 'date' columns to datetime
    start_date = df.date.iloc[0]
    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    start_date_utc = time.mktime(start_date.timetuple())
    end_date = df.date.iloc[-1]
    end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)
    end_date_utc = time.mktime(end_date.timetuple())

    # get eod data
    eod_df = get_dataframe_yf(ticker, start_date_utc, end_date_utc)
    eod_df = eod_df[['low', 'high']]
    eod_df.index = pd.to_datetime(eod_df.index)
    grouped_df = df.groupby(df.index.date).apply(lambda x: x.iloc[-1])

    grouped_df = grouped_df.reset_index().rename({'index': 'date', 'high': 'actual_high', 'low': 'actual_low'}, axis=1)

    # convert 'date' columns to datetime
    df['date'] = pd.to_datetime(df['date'])
    grouped_df['date'] = pd.to_datetime(grouped_df['date'])

    # merge the DataFrames on 'date'
    merged_df = pd.merge(grouped_df, df, on='date', how='left')
    merged_df['date'] = pd.to_datetime(merged_df['date']).dt.date

    # rename columns
    if 'predicted_low_adjusted' in merged_df:
        pred_low_col = 'predicted_low_adjusted'
        pred_high_col = 'predicted_high_adjusted'
    else:
        pred_low_col = 'predicted_low'
        pred_high_col = 'predicted_high'
   
    # avg pct diff
    pct_diff_low = ((merged_df[pred_low_col] - merged_df['actual_low']) / merged_df['actual_low'])
    pct_diff_high = ((merged_df[pred_high_col] - merged_df['actual_high']) / merged_df['actual_high'])
    merged_df['pct_diff_low'] = pct_diff_low * 100
    merged_df['pct_diff_high'] = pct_diff_high * 100

    return merged_df