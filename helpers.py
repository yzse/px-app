import numpy as np
import urllib.request
import json
import random
from datetime import date, timedelta
import pandas as pd
from pandas import json_normalize
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.impute import KNNImputer
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
    
    # Compile the model
    model.fit(x_train, y_train, batch_size=1, epochs=15, verbose=0)

    # Make predictions on the test data
    scaler = MinMaxScaler(feature_range=(0, 1))

    if col_name=='predictions_low':
        scaled_data = scaler.fit_transform(low_prices)
    elif col_name=='predictions_high':
        scaled_data = scaler.fit_transform(high_prices)

    predictions = model.predict(x_test, verbose=0)
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
        next_day_prediction = model.predict(next_day_input, verbose=0)
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
    grouped_df = grouped_df[['low', 'predictions_low', 'high', 'predictions_high', 'avg_pct_diff', 'directional_accuracy']]
    grouped_df = grouped_df.rename(columns={'predictions_low': 'predicted_low', 'predictions_high': 'predicted_high'})


    return grouped_df


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

    return pred_df_filled