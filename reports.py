import streamlit as st
from st_files_connection import FilesConnection
import time
import datetime
from datetime import date, timedelta
import pandas as pd
import s3fs
from helpers import *
pd.set_option('mode.chained_assignment', None)  # Hide SettingWithCopyWarning

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