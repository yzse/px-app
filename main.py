import streamlit as st
from st_files_connection import FilesConnection
import time
import datetime
from datetime import date, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import s3fs
from helpers import *
import seaborn as sns
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
        # ticker with placeholder
        ticker = st.text_input("Ticker:", "", placeholder="e.g. AAPL, TSLA, BTC-USD, ETH-USD")

        # slider
        number_of_days = st.slider("Number of days to lookback:", min_value=20, max_value=365, value=20, step=5)

        submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        # dates
        end_date = datetime.date.today() + timedelta(days=1)
        start_date = end_date - timedelta(days=number_of_days)

        # dataframe
        eod_data_df = get_dataframe_yf(ticker, start_date, end_date)
        low_high_df = eod_data_df.filter(['open', 'low', 'high', 'close', 'volume'])
        
        # chart
        load_chart(low_high_df, ticker)

        ### low/high model
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

        ### historical predictions
        # non-indicator adjustments for group_df
        group_df = get_grouped_df(valid).tail(21)

        if group_df['low'].mean() < 1:
            group_df = group_df.round(4)
        else:
            group_df = group_df.round(2)

        group_df = group_df.applymap(remove_trailing_zeroes)
        
        st.title("Historical Prices Table")
        st.write("Predicted & actual prices for the past 30 days, based on prices from the last 365 days.")
        with st.expander("More about the table"):
            st.write(" - `avg_pct_diff` represents the average difference, as a percentage, between the predicted price and the actual price. It is averaged between the differences between the actual low price & predicted low price, and actual high price & predicted high price.")

            st.write(" - `directional_accuracy` examines whether the predicted price movement matches the actual price movement. It indicates whether the direction of the predicted movement aligns with the direction of the actual movement.")

        # reorder columns
        group_df = group_df[['open', 'high', 'low', 'close', 'predicted_low', 'predicted_high', 'pct_diff_low', 'pct_diff_high', 'predicted_low_direction', 'predicted_high_direction']]

        st.dataframe(group_df)

        ### future predictions

        # end_date = datetime.date.today() - timedelta(days=1)
        next_three_business_days, lows_list, highs_list = predict(end_date - timedelta(days=1), predicted_low, predicted_high)

        last_low = float(group_df.iloc[-1].low)
        last_high = float(group_df.iloc[-1].high)
        last_close = float(group_df.iloc[-1].close)

        pred_df = get_pred_table(next_three_business_days, lows_list, highs_list, last_low, last_high, last_close)

        st.title("Price Prediction Table")
        st.write("Predicted price ranges for the next 3 trading days.")
        with st.expander("More about the model"):
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

        # st.write('Files saved to S3 bucket.')


def show_indicators():

    st.title("Stock Price Prediction for Highs & Lows (+ indicators)")
    
    with st.form(key='my_form_to_submit'):
        ticker = st.text_input("Ticker:", "", placeholder="e.g. AAPL, TSLA, BTC-USD, ETH-USD")
        
        # slider
        number_of_days = st.slider("Number of days to lookback:", min_value=20, max_value=365, value=20, step=5)

        # add indicator selection here
        st.write("Select indicators to include in the lookback model:")
        with st.expander("More about the indicators"):
        
                st.write(" - `VIX:` reflects market uncertainty for the next 30 days based on S&P 500 options. A higher VIX value indicates higher expected market volatility during that period.")

                st.write(" - `Stock Beta:` quantifies a stock's volatility compared to the market, with values over 1.0 indicating greater volatility. The higher the stock beta, the more volatile the predicted price movement will be. Here, the beta is benchmarked against the S&P500.")

                st.write(" - `RSI:` measures the magnitude of recent price changes to evaluate overbought or oversold conditions in the price of a stock. RSI values range from 0 to 100. Traditionally, RSI values over 70 indicate overbought conditions, while values under 30 indicate oversold conditions.")

                st.write(" - `MACD:` is a trend-following momentum indicator that shows the relationship between two moving averages of a stock's price. The MACD is calculated by subtracting the 26-period exponential moving average (EMA) from the 12-period EMA. A nine-day EMA of the MACD, called the signal line, is then plotted on top of the MACD, functioning as a trigger for buy and sell signals.")

                st.write(" - `Bollinger Bands MAVG:`  consists of a Moving Average (MAVG) within Bollinger Bands, used to gauge price volatility and identify potential reversals and breakouts. BB = MAVG ± (2 * SD), where SD = standard deviation of the MAVG.")
        if 'indicators' not in st.session_state:
            st.session_state.indicators = ['VIX', 'sp500', 'beta', 'rsi_14', 'rsi_21', 'macd', 'bb_mavg']
        
        # Display checkboxes for indicators and store their state in session_state
        for i, indicator in enumerate(st.session_state.indicators):
            st.session_state[indicator] = st.checkbox(label=f'{indicator}', key=i)

        indicators_to_keep = [indicator for indicator in st.session_state.indicators if st.session_state[indicator]]

        # get list of indicators to drop based on user selection
        indicators_to_drop = [indicator for indicator in st.session_state.indicators if not st.session_state[indicator]]

        submit_button = st.form_submit_button(label='Submit')

        if submit_button:
        # dates
            end_date = datetime.date.today() +  timedelta(days=1)
            start_date = end_date - timedelta(days=number_of_days)

            # dataframe
            eod_data_df = get_dataframe_yf(ticker, start_date, end_date)
            low_high_df = eod_data_df.filter(['open', 'low', 'high', 'close', 'volume'])

            # get df with indicators
            indicator_df = append_indicators(low_high_df, start_date, end_date)

            # filter df to drop indicators_to_drop
            clean_indicator_df = indicator_df.drop(indicators_to_drop, axis=1)

            st.write("You've selected: `{}`.".format(', '.join(indicators_to_keep)))

            load_chart(low_high_df, ticker)

            # initiate model
            model, scaled_data_low, scaled_data_high, x_train_low, y_train_low, x_test_low, x_train_high, y_train_high, x_test_high = initiate_model(ticker, clean_indicator_df)

            train_size = int(len(scaled_data_low) * 0.8)
            time_steps = 1

            # correlation matrix
            st.subheader("Correlation Matrix")

            # explanation
            with st.expander("More about the correlation matrix"):
                st.write(" - The correlation matrix shows the correlation between the selected indicators and the low & high prices. The closer the value is to 1, the stronger the positive correlation. The closer the value is to -1, the stronger the negative correlation. A value of 0 indicates no correlation.")
                st.write(" - The correlation matrix is calculated using the Pearson correlation coefficient, which measures the linear correlation between two variables X and Y. The coefficient's value ranges from -1 to 1. A value of 1 implies that a linear equation describes the relationship between X and Y perfectly, with all data points lying on a line for which Y increases as X increases. A value of -1 implies that all data points lie on a line for which Y decreases as X increases. A value of 0 implies that there is no linear correlation between the variables.")

            get_correlation_matrix(clean_indicator_df)

            # low & high prediction
            predictions_low_arr, predicted_low = run_model(model, clean_indicator_df, train_size, time_steps, scaled_data_low, x_test_low, x_train_low, y_train_low, 'predictions_low')

            predictions_high_arr, predicted_high = run_model(model, clean_indicator_df, train_size, time_steps, scaled_data_high, x_test_high, x_train_high, y_train_high, 'predictions_high')

            valid = clean_indicator_df[train_size:-1]

            valid['predictions_low'] = predictions_low_arr
            valid['predictions_high'] = predictions_high_arr

            ### historical predictions

            group_df = get_grouped_df(valid).tail(21)

            if group_df['low'].mean() < 1:
                group_df = group_df.round(4)
            else:
                group_df = group_df.round(2)

            group_df = group_df.applymap(remove_trailing_zeroes)
            
            # drop 'date' column
            group_df = group_df.drop('date', axis=1)

            st.write("Predicted & actual prices for the past 30 days, based on prices from the last 365 days, with the selected indicators: `{}`.".format(', '.join(indicators_to_keep)))

            

            st.write("Based on these indicators, the average percentage differences between the predicted prices and actual prices are: ")

            # convert group_df.pct_diff to numeric
            group_df.pct_diff_low = pd.to_numeric(group_df.pct_diff_low, errors='coerce')
            group_df.pct_diff_high = pd.to_numeric(group_df.pct_diff_high, errors='coerce')

            # results
            st.write(f"- Low: `{group_df.pct_diff_low.mean().round(2)}%`")
            st.write(f"- High: `{group_df.pct_diff_high.mean().round(2)}%`")

            st.dataframe(group_df)

            # future predictions

            st.write("Predicted price ranges for the next 3 trading days, based on the selected indicators: `{}`.".format(', '.join(indicators_to_keep)))

            with st.expander("More about the model"):
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

            next_three_business_days, lows_list, highs_list = predict(end_date - timedelta(days=1), predicted_low, predicted_high)

            last_low = float(clean_indicator_df.iloc[-2].low)
            last_high = float(clean_indicator_df.iloc[-2].high)
            last_close = float(clean_indicator_df.iloc[-2].close)

            pred_df = get_pred_table(next_three_business_days, lows_list, highs_list, last_low, last_high, last_close)

            pred_df_adjusted = adjust_pred_table(pred_df)

            st.dataframe(pred_df_adjusted)

            s3 = s3fs.S3FileSystem(anon=False)
            time_t = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"{'myawsbucket-st'}/'streamlit_uploads'/{time_t}"

            with s3.open(f"{path}_indicators_{ticker}_lookback.csv", 'wb') as f:
                group_df.to_csv(f)
            with s3.open(f"{path}_indicators_{ticker}_predictions.csv", 'wb') as f:
                pred_df_adjusted.to_csv(f)

            st.write('Files saved to S3 bucket.')

        

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


