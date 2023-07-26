import streamlit as st
from st_files_connection import FilesConnection
import time
import datetime
from datetime import date, timedelta
import pandas as pd
import s3fs
from helpers import *
pd.set_option('mode.chained_assignment', None)  # Hide SettingWithCopyWarning

# tab1, tab2 = st.tabs(['Application', 'Reports'])

def show_main_page():

    
    # with tab1:

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
        s3 = s3fs.S3FileSystem(anon=False)
        time_t = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"{'myawsbucket-st'}/'streamlit_uploads'/{time_t}"

        with s3.open(f"{path}_{ticker}_lookback.csv", 'wb') as f:
            group_df.to_csv(f)
        with s3.open(f"{path}_{ticker}_predictions.csv", 'wb') as f:
            pred_df.to_csv(f)


def show_main_page_indicators():

    st.title("Stock Price Prediction for Highs & Lows (+ indicators)")

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
        low_high_df = eod_data_df.filter(['low', 'high', 'close'])

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
        group_df = group_df.tail(7) #21

        ##### Vix: pull in vix from last 7 days
        ##### if Vix high -> increase volatility in prediction
        ##### if Vix low -> leave as is
        indicator_df = append_vix_beta(group_df)

        st.title("Indicators")
        st.write(" - VIX: Cboe Volatility Index")
        st.write(" - Stock Beta: Volatility of selected stock")
        
        st.title("Historical Prices Table with Indicators")
        st.write("Predicted & actual prices for the past 30 days, based on prices from the last 365 days, with added indicators.")
        
        st.write(" - `VIX:` reflects market uncertainty for the next 30 days based on S&P 500 options. A higher VIX value indicates higher expected market volatility during that period.")

        st.write(" - `Stock Beta:` quantifies a stock's volatility compared to the market, with values over 1.0 indicating greater volatility. The higher the stock beta, the more volatile the predicted price movement will be. Here, the beta is benchmarked against the S&P500.")

        group_df = group_df[['low', 'predicted_low','high',	'predicted_high','avg_pct_diff','directional_accuracy']]
        st.table(group_df.iloc[1:])

        st.write(" - Added Indicators:")
        st.table(indicator_df)

        ############################# results ###############################

        next_three_business_days, lows_list, highs_list = predict(end_date, predicted_low, predicted_high)

        pred_df = get_pred_table(next_three_business_days, lows_list, highs_list)

        pred_df_adjusted = adjust_pred_table(pred_df)

        st.title("Price Prediction Table with Indicators")
        st.write("Predicted price ranges for the next 3 trading days.")
        st.write(" - The model used here is similar to the previous application. The difference here is that this model takes into account the VIX and stock beta, the modified predicted prices are reflected in the `_adjusted` columns.")
        st.markdown("""
        - Model Parameters:
            - Model: `LSTM`   
            - Neural network layers: `4`
            - Neurons by layer: `128`, `64`, `25`, `1`
            - Optimization Algorithm: `adam`
            - Loss Metric: `mean squared error`
        """)
    
        # st.write(pred_df)

        st.write(pred_df_adjusted)

        
        # upload
        s3 = s3fs.S3FileSystem(anon=False)
        time_t = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"{'myawsbucket-st'}/'streamlit_uploads'/{time_t}"

        with s3.open(f"{path}_{ticker}_lookback.csv", 'wb') as f:
            group_df.to_csv(f)
        with s3.open(f"{path}_{ticker}_predictions.csv", 'wb') as f:
            pred_df.to_csv(f)
        

def show_report_page():
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
        df = conn.read(s_file, input_format="csv", ttl=600)
        df = df.rename({'Unnamed: 0': 'date'}, axis=1)
        st.write(df)