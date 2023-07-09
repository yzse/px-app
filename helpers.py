import numpy as np
import urllib.request
import json
# import time
# import datetime
from datetime import date, timedelta
import pandas as pd
from pandas import json_normalize
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
pd.set_option('mode.chained_assignment', None)  # Hide SettingWithCopyWarning


def get_dataframe(ticker, start_date_utc, end_date_utc):
    url = 'https://eodhistoricaldata.com/api/intraday/{}?api_token=631f8f30266e54.07589191&order=d&interval=1h&fmt=json&from={}&to={}'.format(ticker, start_date_utc, end_date_utc)
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
    model.add(LSTM(64, return_sequences=True, input_shape=(x_train_low.shape[1], 1)))
    model.add(LSTM(32, return_sequences=False))
    # model.add(Dropout(0.1))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model, scaled_data_low, scaled_data_high, x_train_low, y_train_low, x_test_low, y_test_low, x_train_high, y_train_high, x_test_high, y_test_high

def run_model(model, low_high_df, train_size, time_steps, scaled_data, x_test, x_train, y_train, col_name):

    # model
    low_prices = low_high_df['low'].values.reshape(-1, 1)
    high_prices = low_high_df['high'].values.reshape(-1, 1)

    # Scale the data using MinMaxScaler
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # scaled_data_low = scaler.fit_transform(low_prices)
    # scaled_data_high = scaler.fit_transform(high_prices)

    # Compile the model
    model.fit(x_train, y_train, batch_size=32, epochs=15, verbose=0)

    # Make predictions on the test data
    scaler = MinMaxScaler(feature_range=(0, 1))

    if col_name=='predictions_low':
        scaled_data = scaler.fit_transform(low_prices)
    elif col_name=='predictions_high':
        scaled_data = scaler.fit_transform(high_prices)

    predictions = model.predict(x_test, verbose=0)
    predictions = scaler.inverse_transform(predictions)
    # y_test = scaler.inverse_transform([y_test])

    percentage_threshold = 5
    current_price = predictions[-1][0]
    threshold = current_price * (1 - percentage_threshold / 100.0)

    valid = low_high_df[train_size:-1]
    valid[col_name] = predictions

    # Prepare the next day's input data
    next_day_input = np.array([scaled_data[-time_steps:, 0]])
    next_day_input = np.reshape(next_day_input, (next_day_input.shape[0], next_day_input.shape[1], 1))

    # Make predictions for the next 21 values
    predictions_list = []
    for _ in range(21):
        next_day_prediction = model.predict(next_day_input, verbose=0)
        next_day_prediction = np.minimum(next_day_prediction, threshold)
        predictions_list.append(next_day_prediction)
        next_day_input = np.append(next_day_input[:, 1:, :], np.expand_dims(next_day_prediction, axis=1), axis=1)

    # Reshape predictions to match the expected input of inverse_transform
    predictions_list = np.reshape(predictions_list, (len(predictions_list), 1))

    # Inverse transform the predicted prices
    predicted = scaler.inverse_transform(predictions_list)

    return predicted

def is_business_day(date_obj):
    return date_obj.isoweekday() <= 5

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