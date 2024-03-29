import numpy as np
import random
import time
import datetime
from datetime import date, timedelta
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from collections import defaultdict
from statsmodels.tsa.arima.model import ARIMA
import os
import tensorflow as tf
import yfinance as yf
import ta

pd.set_option('mode.chained_assignment', None)

# set seeds
os.environ['PYTHONHASHSEED']=str(1)
tf.random.set_seed(1)
np.random.seed(1)
random.seed(1)

# @st.cache_data(ttl=24*3600, max_entries=3)
def get_dataframe_yf(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)

    # lowercase
    df.index.names = ['datetime']
    df.columns = [x.lower() for x in df.columns]

    return df

def load_chart(low_high_df):
    sma_df = low_high_df.copy()
    st.subheader('Simple Moving Average Chart')
    sma_df['sma_5'] = sma_df['close'].rolling(window=5).mean()
    sma_df['sma_20'] = sma_df['close'].rolling(window=20).mean()
    return st.line_chart(sma_df[['close', 'sma_5', 'sma_20']])

def get_correlation_matrix(df):
    df = df.apply(pd.to_numeric, errors='coerce')
    correlation_matrix = df.corr()
    corr = correlation_matrix[['low', 'high']]
    corr = corr.drop(['open', 'low', 'high', 'close', 'volume'], axis=0)

    # only drop 'date' if exists
    if 'date' in corr.index:
        corr = corr.drop(['date'], axis=0)
    
    return corr

def get_best_ind_params(clean_indicator_df):
    all_corr = get_correlation_matrix(clean_indicator_df).sort_values(by=['low'], ascending=False)
    all_corr = all_corr.reset_index().rename(columns={'index': 'indicator'})
    all_corr['group'] = all_corr['indicator'].apply(lambda x: x.split('_')[0])
    all_corr = all_corr.drop_duplicates(subset='group')
    clean_indicator_df = clean_indicator_df[['open', 'low', 'high', 'close', 'volume'] + all_corr['indicator'].tolist()]
    return clean_indicator_df

def get_highest_corr(clean_indicator_df, corr_pct):

    lookback_periods = [365, 180, 120, 90, 60]
    mean_correlations = {}

    for period in lookback_periods:
        df = clean_indicator_df.tail(period)
        correlation = get_correlation_matrix(df).mean(axis=1).mean()
        mean_correlations[str(period)] = correlation

    top_three_lookbacks = sorted(mean_correlations, key=mean_correlations.get, reverse=True)[:5]
    
    top_three_lookbacks_dfs = []

    for lookback in top_three_lookbacks:
        top_three_lookbacks_dfs.append(clean_indicator_df.tail(int(lookback)))

    result_df = pd.DataFrame(columns=['indicators correlated at > 80%', 'lookback', 'mean correlation'])

    for i in range(len(top_three_lookbacks_dfs)):
        # row names for indicators above 80
        indicators_above_corr_pct = get_correlation_matrix(top_three_lookbacks_dfs[i]).mean(axis=1)[get_correlation_matrix(top_three_lookbacks_dfs[i]).mean(axis=1) >= corr_pct].index.tolist()

        # mean correlation for the indicators_above_80
        indicators_above_corr_pct_mean_corr = get_correlation_matrix(top_three_lookbacks_dfs[i]).mean(axis=1)[get_correlation_matrix(top_three_lookbacks_dfs[i]).mean(axis=1) >= corr_pct].mean()

        # set to 365 if lookback more than 180
        lookback = len(top_three_lookbacks_dfs[i])
        
        adjusted_lookback = 365 if int(lookback) > 180 else int(lookback)

        result_df = result_df._append({
            'indicators correlated at > 80%': indicators_above_corr_pct,
            'lookback': adjusted_lookback,
            'mean correlation': indicators_above_corr_pct_mean_corr
            }, ignore_index=True)

    result_df.dropna(inplace=True)
    result_df.sort_values(by=['mean correlation'], ascending=False, inplace=True)
    result_df.reset_index(drop=True, inplace=True)

    # only show top 3
    result_df = result_df.head(3)

    return result_df

# define LSTM model 
def create_lstm_model(ds):
    model = Sequential(name='lstm_model')
    model.add(LSTM(128, return_sequences=True, input_shape=(ds.shape[1], 1), name='lstm_1'))
    model.add(LSTM(64, return_sequences=False, name='lstm_2'))
    model.add(Dense(25, name='dense_1'))
    model.add(Dense(1, name='dense_2'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def prepare_data(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps - 1):  # Change the range to include the last element
        x_val = data[i:(i + time_steps), 0]
        y_val = data[i + time_steps, 0]
        X.append(x_val)
        y.append(y_val)
    return np.array(X), np.array(y)

# @st.cache_data(ttl=24*3600, max_entries=3)
def initiate_model(low_high_df, best_indicators):

    # model (these are the 'y's)
    low_prices = low_high_df['low'].values.reshape(-1, 1)
    high_prices = low_high_df['high'].values.reshape(-1, 1)

    # extract all columns except 'low' and 'high' for indicators
    indicators = low_high_df[best_indicators]
    
    # normalize price data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data_low = scaler.fit_transform(low_prices)
    scaled_data_high = scaler.fit_transform(high_prices)

    x_train_low, y_train_low = prepare_data(scaled_data_low, time_steps=1)
    x_train_high, y_train_high = prepare_data(scaled_data_high, time_steps=1)

    # model
    model_low = create_lstm_model(x_train_low.reshape(-1, 1))
    model_high = create_lstm_model(x_train_high.reshape(-1, 1))
   
    return model_low, model_high, x_train_low, x_train_high, y_train_low, y_train_high


@st.cache_resource(ttl=24*3600, max_entries=3)
def run_model(_model, df, x_train, y_train, col_name):

    # fit
    x_train = x_train.reshape(-1, 1)
    y_train = y_train.reshape(-1, 1)
    _model.fit(x_train, y_train, batch_size=5, epochs=20, verbose=0)

    low_scaler = MinMaxScaler(feature_range=(0, 1))
    high_scaler = MinMaxScaler(feature_range=(0, 1))

    low_prices = df['low'].values.reshape(-1, 1)
    high_prices = df['high'].values.reshape(-1, 1)

    if col_name=='predictions_low':
        low_scaler.fit_transform(low_prices)

    elif col_name=='predictions_high':
        high_scaler.fit_transform(high_prices)

    # get predicted array
    predicted_array = _model.predict(x_train, verbose=0)

    # inverse transform
    if col_name=='predictions_low':
        predicted_array = low_scaler.inverse_transform(predicted_array)
    elif col_name=='predictions_high':
        predicted_array = high_scaler.inverse_transform(predicted_array)
    
    return predicted_array 


# @st.cache_data(ttl=24*3600, max_entries=3)
def get_grouped_df(df): # turn predictions into table
    df.index = pd.to_datetime(df.index)

    # group and select minimum lows
    df['low'] = pd.to_numeric(df['low'], errors='coerce')
    df['high'] = pd.to_numeric(df['high'], errors='coerce')
    df['date'] = df.index.date
    df['atr'] = abs(df['high'] - df['low'])
    grouped_df = df.groupby(df.index.date).apply(lambda x: x.iloc[-1])

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
    grouped_df['predicted_low_direction'] = grouped_df['diff_low'].apply(lambda x: 'Up' if x > 0 else 'Down')
    grouped_df.drop(columns=['diff_low'], inplace=True)

    # predicted_high_direction
    grouped_df['diff_high'] = grouped_df[pred_high_col] - grouped_df[pred_high_col].shift(1)
    grouped_df['predicted_high_direction'] = grouped_df['diff_high'].apply(lambda x: 'Up' if x > 0 else 'Down')
    grouped_df.drop(columns=['diff_high'], inplace=True)

    # move columns
    grouped_df.insert(2, "predictions_low", grouped_df.pop('predictions_low'))
    grouped_df.insert(4, "predictions_high", grouped_df.pop('predictions_high'))
    
    # rename
    grouped_df = grouped_df.rename(columns={'predictions_low': 'predicted_low', 'predictions_high': 'predicted_high'})

    return grouped_df

def is_business_day(date_obj):
    return date_obj.isoweekday() <= 5

def get_next_3_bus_days(end_date):
    business_days_count = 0
    next_three_business_days = []
    while business_days_count < 3:
        if is_business_day(end_date):
            next_three_business_days.append(end_date)
            business_days_count += 1
        end_date += timedelta(days=1)

    return next_three_business_days

# function to generate a random value within a range
def randomize_value(base, min_percent, max_percent):
    random_factor = random.uniform(min_percent, max_percent)
    return base * (1 + random_factor)

# @st.cache_data(ttl=24*3600, max_entries=3)
def get_pred_table(next_three_business_days, clean_indicator_df):

    clean_indicator_df['atr'] = clean_indicator_df['high'] - clean_indicator_df['low']

    clean_indicator_df = clean_indicator_df.tail(45)
    actual_lows = clean_indicator_df.low.values.tolist()
    actual_highs = clean_indicator_df.high.values.tolist()
    actual_atrs = [high - low for high, low in zip(actual_highs, actual_lows)]

    # get average pct diff from one day's low to the next
    actual_avg_pct_diff = abs(np.mean([((actual_lows[i + 1] - actual_lows[i]) / actual_lows[i]) * 100 for i in range(len(actual_lows) - 1)]))

    low_model = ARIMA(actual_lows, order=(2, 6, 1))
    low_model_fit = low_model.fit()
    next_3_low = low_model_fit.forecast(steps=3)

    # create a Linear Regression model for predicting atrs
    atr_model = ARIMA(actual_atrs, order=(2, 2, 2))
    atr_model_fit = atr_model.fit()
    next_3_atr = atr_model_fit.forecast(steps=3)

    # get pct diff between next_3_low[0] and actual_lows[-1]
    next_pct_diff = abs(((next_3_low[0] - actual_lows[-1]) / actual_lows[-1]) * 100)

    
    # check if next_pct_diff exceeds a threshold
    if next_pct_diff > actual_avg_pct_diff * 1.2:
        next_3_low = [randomize_value(actual_lows[-1], -0.02, 0.02) for _ in range(3)]

    # instantiate df
    pred_df_filled = pd.DataFrame(columns=['predicted_low', 'predicted_atr', 'predicted_high', 'predicted_low_direction', 'predicted_high_direction'])

    # predict next value using linear regression
    pred_df_filled['predicted_low'] = next_3_low
    pred_df_filled['predicted_atr'] = next_3_atr
    pred_df_filled['predicted_high'] = next_3_low + next_3_atr

    pred_df_filled = get_variances(pred_df_filled, next_three_business_days)

    pred_df_filled_dir = get_predicted_direction(pred_df_filled, actual_lows[-1], actual_highs[-1])

    pred_df_filled_dir['variance'] = [1, 2, 3, 1, 2, 3, 1, 2, 3]

    return pred_df_filled_dir

def get_variances(df, next_three_business_days):
    lows_list = df['predicted_low'].values.tolist()
    highs_list = df['predicted_high'].values.tolist()

    rng = np.random.default_rng(123)
    pct_dev = rng.uniform(-2.5, 2.5)

    dates = []
    predicted_lows = []
    predicted_highs = []

    # generate random prices for each date
    for i in range(len(next_three_business_days)):
        date = next_three_business_days[i]

        # add the first row
        dates.append(date)
        predicted_lows.append(lows_list[i])
        predicted_highs.append(highs_list[i])

        # add the second and third rows
        for _ in range(2):
            new_low = predicted_lows[-1] + random.uniform(-pct_dev, pct_dev)
            new_high = predicted_highs[-1] + random.uniform(-pct_dev, pct_dev)

            # append predicted prices for next 3 business days
            dates.append(date)
            predicted_lows.append(new_low)
            predicted_highs.append(new_high)

    res_df = pd.DataFrame({
        'predicted_low': predicted_lows,
        'predicted_high': predicted_highs,
    }, index=dates)

    return res_df

def calculate_averages(pred_df_filled, pred_col, day_count=3):
  averages = []
  for i in range(day_count):
    averages.append(pred_df_filled[pred_col].iloc[i * day_count:(i + 1) * day_count].mean())
  return averages

def get_predicted_direction(pred_df_filled, last_low, last_high):
    # predicted directional column
    pred_low_col = 'predicted_low'
    pred_high_col = 'predicted_high'

    # initialize new columns
    pred_df_filled['predicted_low_direction'] = ''
    pred_df_filled['predicted_high_direction'] = ''

    low_avgs = calculate_averages(pred_df_filled, pred_low_col, 3)
    high_avgs = calculate_averages(pred_df_filled, pred_high_col, 3)

    # set direction based on conditions
    def set_direction(column, values, direction):
        pred_df_filled[column].iloc[values] = direction

    # low direction
    set_direction('predicted_low_direction', slice(0, 3), 'Down' if low_avgs[0] < last_low else 'Up')
    set_direction('predicted_low_direction', slice(3, 6), 'Down' if low_avgs[1] < low_avgs[0] else 'Up')
    set_direction('predicted_low_direction', slice(6, 9), 'Down' if low_avgs[2] < low_avgs[1] else 'Up')

    # high direction
    set_direction('predicted_high_direction', slice(0, 3), 'Down' if high_avgs[0] < last_high else 'Up')
    set_direction('predicted_high_direction', slice(3, 6), 'Down' if high_avgs[1] < high_avgs[0] else 'Up')
    set_direction('predicted_high_direction', slice(6, 9), 'Down' if high_avgs[2] < high_avgs[1] else 'Up')

    return pred_df_filled


def get_accuracy_table(pred_df_filled, clean_indicator_df):
    # accuracy table
    accuracy_df = pred_df_filled.copy()
    accuracy_df['low'] = clean_indicator_df['low']
    accuracy_df['high'] = clean_indicator_df['high']

    # predicted_low_accuracy
    accuracy_df['predicted_low_accuracy'] = accuracy_df.apply(lambda row: 'Correct' if row['predicted_low_direction'] == 'Up' and row['low'] <= row['predicted_low'] or row['predicted_low_direction'] == 'Down' and row['low'] >= row['predicted_low'] else 'Incorrect', axis=1)

    # predicted_high_accuracy
    accuracy_df['predicted_high_accuracy'] = accuracy_df.apply(lambda row: 'Correct' if row['predicted_high_direction'] == 'Up' and row['high'] <= row['predicted_high'] or row['predicted_high_direction'] == 'Down' and row['high'] >= row['predicted_high'] else 'Incorrect', axis=1)

    # move columns
    accuracy_df.insert(2, "predicted_low_accuracy", accuracy_df.pop('predicted_low_accuracy'))
    accuracy_df.insert(4, "predicted_high_accuracy", accuracy_df.pop('predicted_high_accuracy'))

    return accuracy_df

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

# @st.cache_data(ttl=24*3600, max_entries=3)
def append_indicators(df, start_date, end_date, bayesian=False):

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

    if bayesian==False:

        # rsi
        df['rsi_14'] = ta.momentum.RSIIndicator(df['close'], window=14, fillna=True).rsi()
        df['rsi_21'] = ta.momentum.RSIIndicator(df['close'], window=21, fillna=True).rsi()

        # macd
        df['macd'] = ta.trend.macd(df['close'], window_fast=10, window_slow=25, fillna=True)

        # bollinger bands
        indicator_bb = ta.volatility.BollingerBands(close=df["close"], window=20, window_dev=2, fillna=True)
        df['bb_mavg'] = indicator_bb.bollinger_mavg()


    elif bayesian==True: # bayesian optimization

        # params
        rsi_windows = [5, 10, 15, 20, 25, 30]
        macd_windows_fast = [3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        macd_windows_slow = [10, 15, 20, 25, 30, 35, 40, 45, 50, 100]
        bb_windows = [10, 15, 20, 25, 30]
        bb_windows_dev = [1, 2, 3, 4, 5]

        # rsi
        for rsi_window in rsi_windows:
            df[f'rsi_{rsi_window}'] = ta.momentum.RSIIndicator(df['close'], window=rsi_window, fillna=True).rsi()

        # macd
        for macd_window_fast in macd_windows_fast:
            for macd_window_slow in macd_windows_slow:
                df[f'macd_{macd_window_fast}_{macd_window_slow}'] = ta.trend.macd(df['close'], window_fast=macd_window_fast, window_slow=macd_window_slow, fillna=True)

        # bollinger bands
        for bb_window in bb_windows:
            for bb_window_dev in bb_windows_dev:
                indicator_bb = ta.volatility.BollingerBands(close=df["close"], window=bb_window, window_dev=bb_window_dev, fillna=True)
                df[f'bb_{bb_window}_{bb_window_dev}'] = indicator_bb.bollinger_mavg()

    # clean up
    df = df.iloc[1:]
    df = df.applymap(remove_trailing_zeroes)

    # except for first 5 columns, shift dataframe to lag 1 day
    df.iloc[:, 5:] = df.iloc[:, 5:].shift(1)

    return df

def get_atr(df, clean_indicator_df):
    high_col = 'predicted_high'
    low_col = 'predicted_low'

    clean_indicator_df['atr'] = clean_indicator_df['high'] - clean_indicator_df['low']
    last_atr = clean_indicator_df.atr.iloc[-1]

    # set index to new column 'date'
    df['date'] = df.index
    df.reset_index(drop=True, inplace=True)

    # check outliers
    df['predicted_atr'] = abs(df[high_col] - df[low_col])
    df['atr_outlier'] = df['predicted_atr'].apply(lambda x: True if x > last_atr * 2.5 else False)

    def adjust_within_tolerance(previous_value, tolerance=0.05):
        return previous_value + (previous_value * random.uniform(-tolerance, tolerance))

    # adjustments to 'predicted_low' and 'predicted_high' for atr outliers
    for i in range(1, len(df)):
        if df.loc[i, 'atr_outlier']:
            df.loc[i, low_col] = adjust_within_tolerance(df.loc[i - 1, low_col])
            df.loc[i, high_col] = adjust_within_tolerance(df.loc[i - 1, high_col])

    # recalculate atr & highs
    df['predicted_atr'] = abs(df[high_col] - df[low_col])
    df['predicted_high'] = df[low_col] + df['predicted_atr']
    df.insert(2, "predicted_atr", df.pop('predicted_atr'))

    # rounding
    if df.iloc[0, 0] < 1:
        decimals = 4
    elif df.iloc[0, 0] < 20:
        decimals = 3
    else:
        decimals = 2
    df[low_col] = df[low_col].astype(float).round(decimals)
    df[high_col] = df[high_col].astype(float).round(decimals)
    df['predicted_atr'] = df['predicted_atr'].astype(float).round(decimals)

    df = df[['date', 'predicted_low', 'predicted_high', 'predicted_atr', 'predicted_low_direction', 'predicted_high_direction', 'variance']]

    return df

def remove_trailing_zeroes(val):
    if isinstance(val, float):
        return '{:.4f}'.format(val).rstrip('0').rstrip('.')
    return val

def get_perf_df(df, ticker):

    # convert 'date' columns to datetime
    start_date = df.date.iloc[0]
    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    start_date_utc = time.mktime(start_date.timetuple())
    end_date = df.date.iloc[-1]
    end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)
    end_date_utc = time.mktime(end_date.timetuple())

    # get yf eod data
    eod_df = get_dataframe_yf(ticker, start_date_utc, end_date_utc)
    eod_df = eod_df[['low', 'high']]
    eod_df.index = pd.to_datetime(eod_df.index)
    grouped_df = df.groupby(df.index.date).apply(lambda x: x.iloc[-1])

    grouped_df = grouped_df.reset_index().rename({'index': 'date', 'high': 'actual_high', 'low': 'actual_low'}, axis=1)

    # convert 'date' columns to datetime
    df['date'] = pd.to_datetime(df['date'])
    grouped_df['date'] = pd.to_datetime(grouped_df['date'])

    # merge the DataFrames on 'date'
    merged_df = pd.merge(grouped_df, df, on='date', how='left', validate="many_to_many")
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

def show_range_df(df): # show table with ranges of predicted lows and highs
    processed_data = df.groupby('date').agg({
        'predicted_low': ['min', 'max'],
        'predicted_high': ['min', 'max'],
        'predicted_atr': 'mean',
        'predicted_low_direction': lambda x: x.iloc[0],
        'predicted_high_direction': lambda x: x.iloc[0]
    }).reset_index()

    processed_data.columns = ['date', 'predicted_low_min', 'predicted_low_max', 
                              'predicted_high_min', 'predicted_high_max', 'predicted_atr', 
                              'predicted_low_direction', 'predicted_high_direction']

    for i in range(len(processed_data)):
        processed_data.loc[i, 'predicted_low_min'] = randomize_value(processed_data.loc[i, 'predicted_low_min'], -0.0125, 0)
        processed_data.loc[i, 'predicted_low_max'] = randomize_value(processed_data.loc[i, 'predicted_low_max'], 0, 0.0125)
        processed_data.loc[i, 'predicted_high_min'] = randomize_value(processed_data.loc[i, 'predicted_high_min'], -0.0125, 0)
        processed_data.loc[i, 'predicted_high_max'] = randomize_value(processed_data.loc[i, 'predicted_high_max'], 0, 0.0125)

    processed_data['predicted_atr'] = processed_data['predicted_atr'].round(2)

    processed_data['predicted_low_min'] = processed_data['predicted_low_min'].round(2)
    processed_data['predicted_low_max'] = processed_data['predicted_low_max'].round(2)
    processed_data['predicted_high_min'] = processed_data['predicted_high_min'].round(2)
    processed_data['predicted_high_max'] = processed_data['predicted_high_max'].round(2)
    
    processed_data['predicted_low_range'] = processed_data['predicted_low_min'].astype(str) + ' — ' + processed_data['predicted_low_max'].astype(str)

    processed_data['predicted_high_range'] = processed_data['predicted_high_min'].astype(str) + ' — ' + processed_data['predicted_high_max'].astype(str)

    processed_data = processed_data[['date', 'predicted_low_range', 'predicted_high_range', 'predicted_atr', 'predicted_low_direction', 'predicted_high_direction']]

    return processed_data
