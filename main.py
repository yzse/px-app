import streamlit as st
from st_files_connection import FilesConnection
import time
import datetime
from datetime import date, timedelta
import pandas as pd
import s3fs
from helpers import *
pd.set_option('mode.chained_assignment', None)  # Hide SettingWithCopyWarning

tab1, tab2 = st.tabs(['Application', 'Reports'])
def show_main_page():
    with tab1:

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
            

            predictions_low_arr, predicted_low = run_model(model, low_high_df, train_size, time_steps, scaled_data_low, x_test_low, x_train_low, y_train_low, 'predictions_low')

            predictions_high_arr, predicted_high = run_model(model, low_high_df, train_size, time_steps, scaled_data_high, x_test_high, x_train_high, y_train_high, 'predictions_high')

            valid = low_high_df[train_size:-1]
            valid['predictions_low'] = predictions_low_arr
            valid['predictions_high'] = predictions_high_arr

            group_df = get_grouped_df(valid)
            group_df = group_df.tail(21)
            
            st.title("Historical Prices Table")
            st.write("Predicted & actual prices for the past 30 days, based on prices from the last 365 days.")
            
            st.write(" - `avg_pct_diff` represents the average difference, as a percentage, between the predicted price and the actual price. It is averaged between the differences between the actual low price & predicted low price, and actual high price & predicted high price.")

            st.write(" - `directional_accuracy` examines whether the predicted price movement matches the actual price movement. It indicates whether the direction of the predicted movement aligns with the direction of the actual movement.")

            st.table(group_df)

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
            
            # upload
            s3 = s3fs.S3FileSystem(anon=False, refresh=True)
            time_t = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"{'myawsbucket-st'}/'streamlit_uploads'/{time_t}"

            with s3.open(f"{path}_{ticker}_lookback.csv", 'wb') as f:
                group_df.to_csv(f)
            with s3.open(f"{path}_{ticker}_predictions.csv", 'wb') as f:
                pred_df.to_csv(f)

    # fetch
    with tab2:
        conn = st.experimental_connection('s3', type=FilesConnection)
        s3_files = conn.fs.ls(f"{'myawsbucket-st'}/'streamlit_uploads'/", refresh=True)
        s3_file_names = filter_and_reformat_data(s3_files)
        selected_file_name = st.radio("Select a file:", s3_file_names)

        if selected_file_name:
            # get the table from s3
            s_ticker, s_date, s_status = selected_file_name.split(" ")
            s_ticker = s_ticker.lower()
            s_status = s_status.lower()
            s_date = s_date.replace("/", "")
            s_file = find_files_with_substrings(s3_files, [s_ticker, s_date, s_status])

            # display table
            # @st.cache(ttl=24*3600)
            df = conn.read(s_file, input_format="csv", ttl=600)
            df = df.rename({'Unnamed: 0': 'date'}, axis=1)
            st.write(df)
