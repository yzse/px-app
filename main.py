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

        # model
        model, scaled_data_low, scaled_data_high, x_train_low, y_train_low, x_test_low, y_test_low, x_train_high, y_train_high, x_test_high, y_test_high = initiate_model(low_high_df)

        ############################## models ##############################

        train_size = int(len(scaled_data_low) * 0.8)
        time_steps = 1
        
        predicted_low = run_model(model, low_high_df, train_size, time_steps, scaled_data_low, x_test_low, x_train_low, y_train_low, 'predictions_low')

        predicted_high = run_model(model, low_high_df, train_size, time_steps, scaled_data_high, x_test_high, x_train_high, y_train_high, 'predictions_high')
        
        ############################# results ###############################

        next_three_business_days, lows_list, highs_list = predict(end_date, predicted_low, predicted_high)

        low_1 = min(lows_list[0], highs_list[0])
        high_1 = max(lows_list[0], highs_list[0])
        st.write("Predicted range for {} on {}: ".format(ticker, next_three_business_days[0]), low_1, " - ", high_1)

        low_2 = min(lows_list[1], highs_list[1])
        high_2 = max(lows_list[1], highs_list[1])
        st.write("Predicted range for {} on {}: ".format(ticker, next_three_business_days[1]), low_2, " - ", high_2)

        low_3 = min(lows_list[2], highs_list[2])
        high_3 = max(lows_list[2], highs_list[2])
        st.write("Predicted range for {} on {}: ".format(ticker, next_three_business_days[2]), low_3, " - ", high_3)
