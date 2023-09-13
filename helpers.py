import numpy as np
import urllib.request
import json
import random
import time
import datetime
from datetime import date, timedelta
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
# pd.set_option('display.float_format', '{:.4f}'.format)

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
    st.subheader('SMA Chart')
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

def get_highest_corr(clean_indicator_df):

    # Define lookback periods
    lookback_periods = [365, 180, 120, 90, 60]

    # Create a dictionary to store the mean correlations for each lookback period
    mean_correlations = {}

    # Create dataframes and calculate correlations for each lookback period
    for period in lookback_periods:
        df = clean_indicator_df.tail(period)
        correlation = get_correlation_matrix(df).mean(axis=1).mean()
        mean_correlations[str(period)] = correlation

    # Find the lookback period with the highest mean correlation
    top_three_lookbacks = sorted(mean_correlations, key=mean_correlations.get, reverse=True)[:5]

    top_three_lookbacks_dfs = []

    # Iterate through the top three lookback periods and add the corresponding dataframe to the list
    for lookback in top_three_lookbacks:
        top_three_lookbacks_dfs.append(clean_indicator_df.tail(int(lookback)))

    # Create a dataframe to store the selected indicators and their corresponding lookback periods
    result_df = pd.DataFrame(columns=['indicators correlated at > 80%', 'lookback', 'mean correlation'])

    for i in range(len(top_three_lookbacks_dfs)):
        # row names for indicators above 80
        indicators_above_80 = get_correlation_matrix(top_three_lookbacks_dfs[i]).mean(axis=1)[get_correlation_matrix(top_three_lookbacks_dfs[i]).mean(axis=1) >= 0.8].index.tolist()

        # mean correlation for the indicators_above_80
        indicators_above_80_mean_corr = get_correlation_matrix(top_three_lookbacks_dfs[i]).mean(axis=1)[get_correlation_matrix(top_three_lookbacks_dfs[i]).mean(axis=1) >= 0.8].mean()

        # set to 365 if lookback more than 180
        lookback = len(top_three_lookbacks_dfs[i])
        if lookback > 180:
            lookback = 365

        result_df = result_df.append({'indicators correlated at > 80%': indicators_above_80, 'lookback': lookback, 'mean correlation': indicators_above_80_mean_corr}, ignore_index=True)

    result_df.dropna(inplace=True)
    result_df.sort_values(by=['mean correlation'], ascending=False, inplace=True)
    result_df.reset_index(drop=True, inplace=True)

    # only show top 3
    result_df = result_df.head(3)

    return result_df

# define LSTM model 
def create_lstm_model(ds):
    ds = ds.reshape(-1, 1)
    model = Sequential(name='lstm_model')
    model.add(LSTM(128, return_sequences=True, input_shape=(ds.shape[1], 1), name='lstm_1'))
    model.add(LSTM(64, return_sequences=False, name='lstm_2'))
    model.add(Dense(25, name='dense_1'))
    model.add(Dense(1, name='dense_2'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def prepare_data(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps])
        y.append(data[i+time_steps])
    return np.array(X), np.array(y)

@st.cache_data(ttl=24*3600, max_entries=3)
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

    # normalize indicator data
    indicator_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_indicator_data = indicator_scaler.fit_transform(indicators)

    # Combine scaled indicator data with scaled price data
    combined_high = np.concatenate((scaled_data_high, scaled_indicator_data), axis=1)
    combined_low = np.concatenate((scaled_data_low, scaled_indicator_data), axis=1)

    # train test split
    train_data_low, test_data_low = train_test_split(combined_low, test_size=0.2, shuffle=False, random_state=1)
    train_data_high, test_data_high = train_test_split(combined_high, test_size=0.2, shuffle=False, random_state=1)

    time_steps = 1

    # prepare low + high train data
    x_train_low, y_train_low = prepare_data(train_data_low, time_steps)
    x_train_high, y_train_high = prepare_data(train_data_high, time_steps)
    x_test_low, y_test_low = prepare_data(test_data_low, time_steps)
    x_test_high, y_test_high = prepare_data(test_data_high, time_steps)

    # model
    model_low = create_lstm_model(x_train_low)
    model_high = create_lstm_model(x_train_high)
   
    return model_low, model_high, scaled_data_low, scaled_data_high, x_train_low, y_train_low, x_test_low, x_train_high, y_train_high, x_test_high


@st.cache_resource(ttl=24*3600, max_entries=3)
def run_model(_model, df, train_size, x_test, x_train, y_train, col_name):

    # fit
    x_train = x_train.reshape(-1, 1)
    y_train = y_train.reshape(-1, 1)
    x_test = x_test.reshape(-1, 1)
    _model.fit(x_train, y_train, batch_size=1, epochs=15, verbose=0)

    scaler = MinMaxScaler(feature_range=(0, 1))
    low_prices = df['low'].values.reshape(-1, 1)
    high_prices = df['high'].values.reshape(-1, 1)

    if col_name=='predictions_low':
        scaler.fit_transform(low_prices)

    elif col_name=='predictions_high':
        scaler.fit_transform(high_prices)

    # get predicted array
    predicted_array = _model.predict(x_test, verbose=0)
    predicted_array = scaler.inverse_transform(predicted_array)

    # get every 8th value
    predicted_array = predicted_array[::7]
    valid = df[train_size:-1]

    # assign predictions to validation dataframe
    valid[col_name] = predicted_array

    return predicted_array

@st.cache_data(ttl=24*3600, max_entries=3)
def get_grouped_df(df): # turn predictions into table
    df.index = pd.to_datetime(df.index)

    # group and select minimum lows
    df['low'] = pd.to_numeric(df['low'], errors='coerce')
    df['high'] = pd.to_numeric(df['high'], errors='coerce')
    df['date'] = df.index.date
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

def get_next_3_bus_days(end_date):
    business_days_count = 0
    next_three_business_days = []
    while business_days_count < 3:
        if is_business_day(end_date):
            next_three_business_days.append(end_date)
            business_days_count += 1
        end_date += timedelta(days=1)

    return next_three_business_days


@st.cache_data(ttl=24*3600, max_entries=3)
def get_pred_table(next_three_business_days, lows_list, highs_list, clean_indicator_df):

    last_low = float(clean_indicator_df.iloc[-2].low)
    last_high = float(clean_indicator_df.iloc[-2].high)
    

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
    # mask = res['predicted_low'].round(1) == res['predicted_high'].round(1)

    # # Compute the random deviations based on 5% of 'predicted_high'
    # deviations = res.loc[mask, 'predicted_high'] * np.random.uniform(-0.25, 0.125, size=mask.sum())

    # # Add the random deviations to 'predicted_high' for rows where the mask is True
    # res.loc[mask, 'predicted_high'] += deviations

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

    # damping deviation
    pred_df_filled['predicted_low'] = last_low + (pred_df_filled['predicted_low'] - last_low) * 0.15
    pred_df_filled['predicted_high'] = last_high + (pred_df_filled['predicted_high'] - last_high) * 0.15
    
    # predicted_atr
    # pred_df_filled['predicted_atr'] = pred_df_filled.apply(calculate_atr, axis=1, last_close=last_close, pred_low_col=pred_low_col, pred_high_col=pred_high_col)

    # # move columns
    # pred_df_filled.insert(2, "predicted_atr", pred_df_filled.pop('predicted_atr'))

    # make sure predicted_low < predicted_high
    pred_df_filled.loc[pred_df_filled['predicted_low'] > pred_df_filled['predicted_high'], ['predicted_low', 'predicted_high']] = pred_df_filled.loc[pred_df_filled['predicted_low'] > pred_df_filled['predicted_high'], ['predicted_high', 'predicted_low']].values

    return pred_df_filled

def calculate_tr(row, last_close):
    predicted_high = pd.to_numeric(row['predicted_high_adjusted'])
    predicted_low = pd.to_numeric(row['predicted_low_adjusted'])
    last_close = pd.to_numeric(last_close)
    
    return max(predicted_high - predicted_low, abs(predicted_high - last_close), abs(predicted_low - last_close))

def get_atr(pred_df_adjusted, clean_indicator_df):
    # Calculate last_close from clean_indicator_df
    last_close = clean_indicator_df.iloc[-2].close

    # Ensure that the columns contain numeric values
    pred_df_adjusted['predicted_high_adjusted'] = pd.to_numeric(pred_df_adjusted['predicted_high_adjusted'])
    pred_df_adjusted['predicted_low_adjusted'] = pd.to_numeric(pred_df_adjusted['predicted_low_adjusted'])
    
    # Calculate predicted_atr
    pred_df_adjusted['predicted_atr'] = pred_df_adjusted.apply(calculate_tr, axis=1, last_close=last_close)

    # move atr to 3rd col
    pred_df_adjusted.insert(2, "predicted_atr", pred_df_adjusted.pop('predicted_atr'))

    return pred_df_adjusted


def get_accuracy_table(pred_df_filled, clean_indicator_df):
    # accuracy table
    accuracy_df = pred_df_filled.copy()
    accuracy_df['low'] = clean_indicator_df['low']
    accuracy_df['high'] = clean_indicator_df['high']

    # predicted_low_accuracy
    accuracy_df['predicted_low_accuracy'] = accuracy_df.apply(lambda row: 'Correct' if row['predicted_low_direction'] == 'Increase' and row['low'] <= row['predicted_low'] or row['predicted_low_direction'] == 'Decrease' and row['low'] >= row['predicted_low'] else 'Incorrect', axis=1)

    # predicted_high_accuracy
    accuracy_df['predicted_high_accuracy'] = accuracy_df.apply(lambda row: 'Correct' if row['predicted_high_direction'] == 'Increase' and row['high'] <= row['predicted_high'] or row['predicted_high_direction'] == 'Decrease' and row['high'] >= row['predicted_high'] else 'Incorrect', axis=1)

    # move columns
    accuracy_df.insert(2, "predicted_low_accuracy", accuracy_df.pop('predicted_low_accuracy'))
    accuracy_df.insert(4, "predicted_high_accuracy", accuracy_df.pop('predicted_high_accuracy'))

    return accuracy_df

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
    # atr_adjustments = np.random.normal(loc=df['predicted_atr'], scale=df['predicted_atr'] * 0.01)
    
    # clip the adjustments to be within the tolerance range
    low_adjusted = np.clip(low_adjustments, a_min=None, a_max=high_adjustments)
    
    # adjust 'low_adjusted' and 'high_adjustments' if they are identical
    mask = low_adjusted == high_adjustments
    low_adjusted[mask] -= 0.001 
    high_adjustments[mask] += 0.001 
    
    # set adjusted prices in the DataFrame
    df['predicted_low_adjusted'] = low_adjusted
    df['predicted_high_adjusted'] = high_adjustments
    # df['predicted_atr_adjusted'] = atr_adjustments

    # reduce variation
    smoothing_factor = 0.1
    mean_val = (df["predicted_low_adjusted"] + df["predicted_high_adjusted"]) / 2
    diff = (df["predicted_high_adjusted"] - df["predicted_low_adjusted"]) * smoothing_factor
    df["predicted_high_adjusted"] = mean_val + diff
    df["predicted_low_adjusted"] = mean_val - diff

    df = df[['predicted_low_adjusted', 'predicted_high_adjusted', 'predicted_low_direction', 'predicted_high_direction']]

    df['variance'] = [1, 2, 3, 1, 2, 3, 1, 2, 3]

    # round to 4 significant figures
    df = df.applymap(remove_trailing_zeroes)


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