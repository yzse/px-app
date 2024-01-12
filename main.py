import streamlit as st
from st_files_connection import FilesConnection
import time
import datetime
from datetime import date, timedelta
import pandas as pd
import s3fs
from helpers import *
pd.set_option('mode.chained_assignment', None)

# set seeds
os.environ['PYTHONHASHSEED']=str(1)
tf.random.set_seed(1)
np.random.seed(1)
random.seed(1)


def show_indicators():

    st.title("Stock Price Prediction for Highs & Lows (+ indicators)")

    with st.expander("Available Indicators"):

        st.write(" - `VIX:` reflects market uncertainty for the next 30 days based on S&P 500 options. A higher VIX value indicates higher expected market volatility during that period.")

        st.write(" - `Stock Beta:` quantifies a stock's volatility compared to the market, with values over 1.0 indicating greater volatility. The higher the stock beta, the more volatile the predicted price movement will be. Here, the beta is benchmarked against the S&P500.")

        st.write(" - `RSI:` measures the magnitude of recent price changes to evaluate overbought or oversold conditions in the price of a stock. RSI values range from 0 to 100. Traditionally, RSI values over 70 indicate overbought conditions, while values under 30 indicate oversold conditions.")

        st.write(" - `MACD:` is a trend-following momentum indicator that shows the relationship between two moving averages of a stock's price. The MACD is calculated by subtracting the 26-period exponential moving average (EMA) from the 12-period EMA. A nine-day EMA of the MACD, called the signal line, is then plotted on top of the MACD, functioning as a trigger for buy and sell signals.")

        st.write(" - `Bollinger Bands MAVG:`  consists of a Moving Average (MAVG) within Bollinger Bands, used to gauge price volatility and identify potential reversals and breakouts. BB = MAVG ± (2 * SD), where SD = standard deviation of the MAVG.")
    
    with st.form(key='my_form_to_submit'):
        ticker = st.text_input("Ticker:", "", placeholder="e.g. AAPL, TSLA, BTC-USD, ETH-USD")
        
        # slider
        number_of_days = 365

        submit_button = st.form_submit_button(label='Submit')

        if submit_button:
            # runtime
            start_time = time.time()

            # dates
            end_date = datetime.date.today() + timedelta(days=1)
            start_date = end_date - timedelta(days=number_of_days)

            # dataframe
            eod_data_df = get_dataframe_yf(ticker, start_date, end_date)
            low_high_df = eod_data_df.filter(['open', 'low', 'high', 'close', 'volume'])

            # get df with indicators
            clean_indicator_df = append_indicators(low_high_df, start_date, end_date, bayesian=True)
            clean_indicator_df = clean_indicator_df.apply(pd.to_numeric, errors='coerce')
            

            if clean_indicator_df.isnull().values.any():
                # impute missing values
                clean_indicator_df = clean_indicator_df.fillna(method='ffill')
                clean_indicator_df = clean_indicator_df.fillna(method='bfill')

            load_chart(low_high_df)

            # correlation matrix
            st.subheader("Highest Correlation with Low & High Prices")

            # explanation
            with st.expander("More about the correlation matrix"):
                st.write(" - The correlation matrix shows the correlation between the selected indicators and the low & high prices. The closer the value is to 1, the stronger the positive correlation. The closer the value is to -1, the stronger the negative correlation. A value of 0 indicates no correlation.")
                st.write(" - The correlation matrix is calculated using the Pearson correlation coefficient, which measures the linear correlation between two variables X and Y. The coefficient's value ranges from -1 to 1. A value of 1 implies that a linear equation describes the relationship between X and Y perfectly, with all data points lying on a line for which Y increases as X increases. A value of -1 implies that all data points lie on a line for which Y decreases as X increases. A value of 0 implies that there is no linear correlation between the variables.")
                st.write(" - The column 'indicators correlated at > 80%' shows the indicators that have a correlation of over 80% with the low & high prices. These indicators are used to train the model. The number of days to lookback is also selected based on the highest correlation with the low & high prices. The mean correlation is calculated by averaging the correlation between the low price and the high price.")

            clean_indicator_df = get_best_ind_params(clean_indicator_df)
            best_corr_df = get_highest_corr(clean_indicator_df, corr_pct=0.8)
            best_indicators, best_lookback = best_corr_df.iloc[0][0], best_corr_df.iloc[0][1]

            model_low, model_high, x_train_low, y_train_low, x_train_high, y_train_high = initiate_model(clean_indicator_df, best_indicators)

            # highlight first row
            st.dataframe(
                best_corr_df.style.applymap(
                    lambda _: "background-color: #82C79F;", subset=([0], slice(None))
                )
            )

            # low & high prediction
            predictions_low_arr = run_model(model_low, clean_indicator_df, x_train_low, y_train_low, 'predictions_low')
            predictions_high_arr = run_model(model_high, clean_indicator_df, x_train_high, y_train_high, 'predictions_high')
            
            valid = clean_indicator_df[:-2]
            valid['predictions_low'] = predictions_low_arr
            valid['predictions_high'] = predictions_high_arr

            # historical predictions
            group_df = get_grouped_df(valid).tail(21)
            group_df = group_df.applymap(remove_trailing_zeroes)
            
            # drop 'date' column
            group_df = group_df.drop('date', axis=1)

            # convert group_df.pct_diff to numeric
            group_df.pct_diff_low = pd.to_numeric(group_df.pct_diff_low, errors='coerce')
            group_df.pct_diff_high = pd.to_numeric(group_df.pct_diff_high, errors='coerce')

            # reorder columns
            group_df_show = group_df[['low', 'predicted_low', 'high', 'predicted_high', 'atr'] + best_indicators + ['pct_diff_low', 'pct_diff_high', 'predicted_low_direction', 'predicted_high_direction']]

            st.write("The table below shows the historically predicted prices for `${}` using the model with these parameters, displaying the last 30 days.".format(ticker.upper()))

            st.dataframe(group_df_show)
            
            # store the accuracy scores
            accuracy_scores_df = pd.DataFrame({
                'Price Point': ['Low', 'High'],
                'Avg % Diff': [round(group_df.pct_diff_low.mean(), 2), round(group_df.pct_diff_high.mean(), 2)]
            })

            # accuracy scores
            st.write("Accuracy scores for between predicted & actual prices, calculated by averaging the differences between predicted & actual prices.")
            st.dataframe(accuracy_scores_df)

            # future predictions
            st.write("Predicted price ranges for the next 3 trading days, based on the best performing indicators and lookback period. Predicted directions are calculated using the average of 3 variances and the previous day's prices.")

            with st.expander("More about the model"):
                st.write(" - The model used here is the Long Short-Term Memory (LSTM) model. It is a  neural network architecture used for time series forecasting. It leverages historical price data to capture complex patterns and dependencies over time. By processing sequential data, LSTM models can learn from the historical price movements of stocks, identifying trends and patterns that may impact future prices.")
                st.write(" - `predicted_atr` represents the average of true ranges over the specified period with a focus on daily measurements. This metric captures volatility by accounting for any gaps in price movement on a daily basis.")
                st.write(" - For each day of prediction, the model provides a predicted low price, a predicted high price, and a predicted average true range (atr); along with a list of predicted variances.  3 of the variances are selected and applied to the final predicted output.")
                st.markdown("""
                - Model Parameters:
                    - Model: `LSTM`   
                    - Neural network layers: `4`
                    - Neurons by layer: `128`, `64`, `25`, `1`
                    - Optimization Algorithm: `adam`
                    - Loss Metric: `mean squared error`
                """)

            # predict low & atr, set high as low + atr
            pred_df = get_pred_table(get_next_3_bus_days(end_date), clean_indicator_df)

            pred_df_atr = get_atr(pred_df, clean_indicator_df)

            # st.dataframe(pred_df_atr)

            st.write("Predicted price ranges for the next 3 trading days.")
            st.dataframe(show_range_df(pred_df_atr))

            s3 = s3fs.S3FileSystem(anon=False)
            time_t = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"{'myawsbucket-st'}/'streamlit_uploads'/{time_t}"

            with s3.open(f"{path}_indicators_{ticker}_lookback.csv", 'wb') as f:
                group_df_show.to_csv(f)
            with s3.open(f"{path}_indicators_{ticker}_predictions.csv", 'wb') as f:
                pred_df_atr.to_csv(f)

            # runtime in minutes
            end_time = time.time()
            runtime = round((end_time - start_time)/60, 2)
            st.write("Runtime: {} minutes.".format(runtime))
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
                df = df.drop('Unnamed: 0', axis=1)
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


