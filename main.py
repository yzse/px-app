import streamlit as st
import time
import datetime
from datetime import date, timedelta
import pandas as pd
from helpers import *
pd.set_option('mode.chained_assignment', None)  # Hide SettingWithCopyWarning

def show_main_page():

    st.title("Stock Price Prediction for Highs & Lows")

    # ticker
    with st.form(key='my_form_to_submit'):
        ticker = st.text_input("Ticker:", "")
        submit_button = st.form_submit_button(label='Submit')

    if submit_button:

        # dates
        end_date = datetime.date.today()
        end_date_utc = time.mktime(end_date.timetuple())
        start_date = end_date - timedelta(days=365)
        start_date_utc = time.mktime(start_date.timetuple())

        # dataframe
        eod_data_df = get_dataframe(ticker, start_date_utc, end_date_utc)
        low_high_df = eod_data_df.filter(['low', 'high'])
        # grouped_df = get_grouped_df(low_high_df)

        # model
        model, scaled_data_low, scaled_data_high, x_train_low, y_train_low, x_test_low, y_test_low, x_train_high, y_train_high, x_test_high, y_test_high = initiate_model(low_high_df)

        ############################## models ##############################

        train_size = int(len(scaled_data_low) * 0.8)
        time_steps = 1
        

        predictions_low_arr, predicted_low = run_model(model, low_high_df, train_size, time_steps, scaled_data_low, x_test_low, x_train_low, y_train_low, 'predictions_low')

        predictions_high_arr, predicted_high = run_model(model, low_high_df, train_size, time_steps, scaled_data_high, x_test_high, x_train_high, y_train_high, 'predictions_high')

        valid = low_high_df[train_size:-1]
        valid['predictions_low'] = predictions_low_arr
        valid['predictions_high'] = predictions_high_arr

        group_df = get_grouped_df(valid)
        
        st.title("Historical Prices Table")
        st.write("Predicted & actual prices for the past 30 days, based on prices from the last 365 days.")
        
        st.write(" - `avg_pct_diff` represents the average difference, as a percentage, between the predicted price and the actual price. It is averaged between the differences between the actual low price & predicted low price, and actual high price & predicted high price.")

        st.write(" - `directional_accuracy` examines whether the predicted price movement matches the actual price movement. It indicates whether the direction of the predicted movement aligns with the direction of the actual movement.")

        st.table(group_df.tail(21))

        ############################# results ###############################

        next_three_business_days, lows_list, highs_list = predict(end_date, predicted_low, predicted_high)

        pred_df = get_pred_table(next_three_business_days, lows_list, highs_list)

        st.title("Price Prediction Table")
        st.write("Predicted price ranges for the next 3 trading days.")
        st.write(" - The model used here is the Long Short-Term Memory (LSTM) model. It is a  neural network architecture used for time series forecasting. It leverages historical price data to capture complex patterns and dependencies over time. By processing sequential data, LSTM models can learn from the historical price movements of stocks, identifying trends and patterns that may impact future prices.")
        st.markdown("""
        - Model Parameters:
            - Model: `LSTM`   
            - Neural network layers: `4`
            - Neurons by layer: `128`, `64`, `25`, `1`
            - Optimization Algorithm: `adam`
            - Loss Metric: `mean squared error`
        """)
    
        st.write(pred_df)