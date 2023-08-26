import streamlit as st
from st_files_connection import FilesConnection
import time
import datetime
from datetime import date, timedelta
import pandas as pd
import s3fs
from helpers import *
pd.set_option('mode.chained_assignment', None)
pd.set_option('display.float_format', '{:.2f}'.format)

# set seeds
os.environ['PYTHONHASHSEED']=str(1)
tf.random.set_seed(1)
np.random.seed(1)
random.seed(1)

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
        if ticker.endswith('.cc'):
            start_date = end_date - timedelta(days=90)
        else:
            start_date = end_date - timedelta(days=365)
        start_date_utc = time.mktime(start_date.timetuple())

        # dataframe
        eod_data_df = get_dataframe(ticker, start_date_utc, end_date_utc)
        low_high_df = eod_data_df.filter(['open', 'low', 'high', 'close', 'volume'])
        
        # chart
        load_chart(low_high_df, ticker)

        if ticker.endswith('.cc'):
            low_high_df = crypto_date_filter(low_high_df)
            
        ############################## low/high model ##############################

        # initiate model
        model, scaled_data_low, scaled_data_high, x_train_low, y_train_low, x_test_low, x_train_high, y_train_high, x_test_high = initiate_model(ticker, low_high_df)

        train_size = int(len(scaled_data_low) * 0.8)
        time_steps = 1

        # low & high prediction
        predictions_low_arr, predicted_low = run_model(model, low_high_df, train_size, time_steps, scaled_data_low, x_test_low, x_train_low, y_train_low, 'predictions_low')
        

        predictions_high_arr, predicted_high = run_model(model, low_high_df, train_size, time_steps, scaled_data_high, x_test_high, x_train_high, y_train_high, 'predictions_high')

        valid = low_high_df[train_size:-1]

        valid['predictions_low'] = predictions_low_arr
        valid['predictions_high'] = predictions_high_arr

        ############################### results dataframe ##############################

        group_df = get_grouped_df(valid)

        # non-indicator adjustments for group_df
        group_df = group_df.tail(21)

        if group_df['low'].mean() < 1:
            group_df = group_df.round(4)
        else:
            group_df = group_df.round(2)

        group_df = group_df.applymap(remove_trailing_zeroes)
        
        st.title("Historical Prices Table")
        st.write("Predicted & actual prices for the past 30 days, based on prices from the last 365 days.")
        
        st.write(" - `avg_pct_diff` represents the average difference, as a percentage, between the predicted price and the actual price. It is averaged between the differences between the actual low price & predicted low price, and actual high price & predicted high price.")

        st.write(" - `directional_accuracy` examines whether the predicted price movement matches the actual price movement. It indicates whether the direction of the predicted movement aligns with the direction of the actual movement.")


        st.dataframe(group_df)


        ############################# results ###############################

        next_three_business_days, lows_list, highs_list = predict(end_date, predicted_low, predicted_high)

        last_low = float(group_df.iloc[-1].low)
        last_high = float(group_df.iloc[-1].high)
        last_close = float(group_df.iloc[-1].close)

        pred_df = get_pred_table(next_three_business_days, lows_list, highs_list, last_low, last_high, last_close)

        st.title("Price Prediction Table")
        st.write("Predicted price ranges for the next 3 trading days.")
        st.write(" - The model used here is the Long Short-Term Memory (LSTM) model. It is a  neural network architecture used for time series forecasting. It leverages historical price data to capture complex patterns and dependencies over time. By processing sequential data, LSTM models can learn from the historical price movements of stocks, identifying trends and patterns that may impact future prices.")
        st.write(" - `predicted_atr` represents the average of true ranges over the specified period with a focus on daily measurements. This metric captures volatility by accounting for any gaps in price movement on a daily basis. `ATR = (ATR_previous * (n - 1) + TR_current`")
        st.write(" - For each day of prediction, the model provides a predicted low price, a predicted high price, and a predicted average true range (atr); along with a list of predicted variances.  3 of the variances are selected and applied to the final predicted output.")
        st.markdown("""
        - Model Parameters:
            - Model: `LSTM`   
            - Neural network layers: `4`
            - Neurons by layer: `128`, `64`, `25`, `1`
            - Optimization Algorithm: `adam`
            - Loss Metric: `mean squared error`
        """)
        
        st.dataframe(pred_df)
        
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
        if ticker.endswith('.cc'):
            start_date = end_date - timedelta(days=90)
        else:
            start_date = end_date - timedelta(days=365)
        start_date_utc = time.mktime(start_date.timetuple())

        # dataframe
        eod_data_df = get_dataframe(ticker, start_date_utc, end_date_utc)
        low_high_df = eod_data_df.filter(['open', 'low', 'high', 'close', 'volume'])

        # chart
        load_chart(low_high_df, ticker)

        if ticker.endswith('.cc'):
            low_high_df = crypto_date_filter(low_high_df)

        ############################## low/high model ##############################
        
        # initiate model
        model, scaled_data_low, scaled_data_high, x_train_low, y_train_low, x_test_low, x_train_high, y_train_high, x_test_high = initiate_model(ticker, low_high_df)

        train_size = int(len(scaled_data_low) * 0.8)
        time_steps = 1

        # low & high prediction
        predictions_low_arr, predicted_low = run_model(model, low_high_df, train_size, time_steps, scaled_data_low, x_test_low, x_train_low, y_train_low, 'predictions_low')

        predictions_high_arr, predicted_high = run_model(model, low_high_df, train_size, time_steps, scaled_data_high, x_test_high, x_train_high, y_train_high, 'predictions_high')

        valid = low_high_df[train_size:-1]

        valid['predictions_low'] = predictions_low_arr
        valid['predictions_high'] = predictions_high_arr

        ############################### results dataframe ##############################

        group_df = get_grouped_df(valid)

        # st.write(valid.columns)

        # indicator adjustments for group_df
        group_df = group_df.tail(7)

        # indicators
        last_close = float(group_df.iloc[-1].close)
        indicator_df = append_vix_beta(group_df, last_close)
        indicator_df = adjust_indicator_table(indicator_df)
        indicator_df['predicted_high_adjusted'] = pd.to_numeric(indicator_df['predicted_high_adjusted'], errors='coerce')


        if indicator_df['predicted_high_adjusted'].mean() < 1:
            indicator_df = indicator_df.round(4)
        else:
            indicator_df = indicator_df.round(2)

        group_df = group_df[['low', 'predicted_low','high',	'predicted_high', 'pct_diff_low', 'pct_diff_high', 'predicted_low_direction', 'predicted_high_direction']]

        st.title("Historical Prices Table with Indicators")
        st.write("Predicted & actual prices for the past 30 days, based on prices from the last 365 days, with added indicators.")
        
        st.write(" - `VIX:` reflects market uncertainty for the next 30 days based on S&P 500 options. A higher VIX value indicates higher expected market volatility during that period.")

        st.write(" - `Stock Beta:` quantifies a stock's volatility compared to the market, with values over 1.0 indicating greater volatility. The higher the stock beta, the more volatile the predicted price movement will be. Here, the beta is benchmarked against the S&P500.")

        st.write(" - `RSI:` measures the magnitude of recent price changes to evaluate overbought or oversold conditions in the price of a stock. RSI values range from 0 to 100. Traditionally, RSI values over 70 indicate overbought conditions, while values under 30 indicate oversold conditions.")

        st.write(" - `MACD:` is a trend-following momentum indicator that shows the relationship between two moving averages of a stock's price. The MACD is calculated by subtracting the 26-period exponential moving average (EMA) from the 12-period EMA. A nine-day EMA of the MACD, called the signal line, is then plotted on top of the MACD, functioning as a trigger for buy and sell signals.")

        st.write(" - `Bollinger Bands MAVG:`  consistings of a Moving Average (MAVG) within Bollinger Bands, used to gauge price volatility and identify potential reversals and breakouts. BB = MAVG ± (2 * SD), where SD = standard deviation of the MAVG.")

        group_df = group_df.applymap(remove_trailing_zeroes)

        clean_indicator_df = indicator_df.drop(['close', 'open', 'volume', 'predicted_low_adjusted', 'predicted_high_adjusted', 'pct_diff_low_adjusted', 'pct_diff_high_adjusted', 'predicted_atr_adjusted'], axis=1)
        
        st.dataframe(clean_indicator_df)

        ############################# results ###############################

        next_three_business_days, lows_list, highs_list = predict(end_date, predicted_low, predicted_high)

        last_low = float(indicator_df.iloc[-1].low)
        last_high = float(indicator_df.iloc[-1].high)
        last_close = float(indicator_df.iloc[-1].close)

        pred_df = get_pred_table(next_three_business_days, lows_list, highs_list, last_low, last_high, last_close)

        pred_df_adjusted = adjust_pred_table(pred_df)


        st.title("Price Prediction Table with Indicators")
        st.write("Predicted price ranges for the next 3 trading days.")
        st.write(" - The model used here is similar to the previous application. The difference here is that this model takes into account the VIX and stock beta, the modified predicted prices are reflected in the `_adjusted` columns.")
        st.write(" - `predicted_atr` represents the average of true ranges over the specified period with a focus on daily measurements. This metric captures volatility by accounting for any gaps in price movement on a daily basis. `ATR = (ATR_previous * (n - 1) + TR_current`")
        st.write(" - For each day of prediction, the model provides a predicted low price and a predicted high price, along with a list of predicted variances.  3 of the variances are selected and applied to the final predicted output.")
        st.markdown("""
        - Model Parameters:
            - Model: `LSTM`   
            - Neural network layers: `4`
            - Neurons by layer: `128`, `64`, `25`, `1`
            - Optimization Algorithm: `adam`
            - Loss Metric: `mean squared error`
        """)


        # Initialize the session state
        if 'indicators' not in st.session_state:
            st.session_state.indicators = ['vix', 'beta', 'rsi_14', 'rsi_21', 'macd', 'bb_mavg']

        # Display checkboxes for indicators and store their state in session_state
        for i, indicator in enumerate(st.session_state.indicators):
            st.session_state[indicator] = st.checkbox(label=f'{indicator}', key=i)

        # Add run button
        run_button = st.button(label='Run Model')

        # Only run if at least one indicator is selected
        if run_button:
            # Create a list of selected indicators
            selected_indicators = [indicator for indicator in st.session_state.indicators if st.session_state[indicator]]

            if selected_indicators:
                # Generate some example data (replace this with your actual data generation logic)

                # Upload data to S3
                s3 = s3fs.S3FileSystem(anon=False)
                time_t = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                path = f"myawsbucket-st/streamlit_uploads/{time_t}"

                with s3.open(f"{path}_indicators_{ticker}_lookback.csv", 'wb') as f:
                    clean_indicator_df.to_csv(f)
                with s3.open(f"{path}_indicators_{ticker}_predictions.csv", 'wb') as f:
                    pred_df_adjusted.to_csv(f)

        

def show_report_page():
    conn = st.experimental_connection('s3', type=FilesConnection)
    s3_files = conn.fs.ls(f"{'myawsbucket-st'}/'streamlit_uploads'/", refresh=True)

    s3_tickers, s3_file_names = filter_and_reformat_data(s3_files)

    option = st.selectbox("Select a ticker:", sorted(s3_tickers))

    if option:
        selected = [s for s in s3_file_names if option in s]
        selected_report = st.radio('Select a report: ', selected)

        if selected_report:
            # get the table from s3

            splitted_names = selected_report.split(" ")
            if len(splitted_names) == 4:
                s_ticker, s_date, s_status, s_ind = splitted_names
            else:
                s_ticker, s_date, s_status = splitted_names
                s_ind = ''

            s_ticker = s_ticker.lower()
            s_status = s_status.lower()
            s_ind = s_ind.lower()
            s_date = s_date.replace("/", "")

            if s_ind == '':
                s_file = find_files_with_substrings(s3_files, [s_ticker, s_date, s_status, s_ind])
            else:
                s_file = find_files_with_substrings(s3_files, [s_ticker, s_date, s_status])

            
            # display table
            try:
                df = conn.read(s_file, input_format="csv", ttl=600)
                df = df.rename({'Unnamed: 0': 'date'}, axis=1)
                st.dataframe(df)
            
                if 'predictions' in s_file:

                    get_performance_button = st.button(label='Get Performance')

                    last_report_date = datetime.datetime.strptime(df.date.iloc[-1], '%Y-%m-%d') + timedelta(days=1)
                    
                    today_date = datetime.datetime.today()

                    business_days_diff = np.busday_count(last_report_date.date(), today_date.date())

                    perf_days_check = business_days_diff >= 1
                    
                    if get_performance_button:

                        if perf_days_check==0:

                            st.write('Please wait 1 trading day after the last predicted day of the report to track performance.')

                        else:
                            
                            perf_df = get_perf_df(df, s_ticker)
                            st.dataframe(perf_df)

            except AttributeError:
                st.write('File not found.')
