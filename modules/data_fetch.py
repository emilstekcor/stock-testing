# modules/data_fetch.py
import streamlit as st
import pandas as pd
import logging
import traceback
from datetime import datetime
from typing import Tuple

logger = logging.getLogger(__name__)
MAX_HISTORY_LENGTH_YEARS = 6

def fetch_data(ticker: str, start_date: datetime, end_date: datetime, column_names:Tuple[str,...] = ('Open', 'Close', 'High', 'Low'), adjust_close: bool = True, data_source: str = 'yfinance') -> pd.DataFrame:
    """
    Fetches stock data, with support for second-level data from other providers.
    """
    try:
        if data_source == 'yfinance':
             import yfinance as yf
             data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        elif data_source == 'my_second_level_provider':
            # Placeholder: Replace with your second-level data API call
            data = _fetch_second_level_data(ticker, start_date, end_date)
        else:
             raise ValueError(f"Invalid data source: {data_source}")

    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        logger.error(f"Error fetching data for {ticker}: {e} ", exc_info=True)
        return pd.DataFrame()
    if data.empty:
        st.error(f"No data fetched for {ticker}.")
        logger.error(f"No data fetched for {ticker}.")
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    if adjust_close and data_source == 'yfinance':
      alt_map = {'Close':'Adj Close'}
      for col, alt in alt_map.items():
        if col not in data.columns and alt in data.columns:
            data.rename(columns={alt: col}, inplace=True)
    if not set(column_names).issubset(data.columns):
         st.warning(f"Missing needed columns; data fetched contains: {data.columns}")
         logger.warning(f"Missing needed columns for data fetch in config : {column_names}  -> columns present {data.columns}")
         return pd.DataFrame()
    data.dropna(subset=list(column_names), inplace=True)
    return data

def _fetch_second_level_data(ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Placeholder function to simulate fetching second-level data.
    Replace this with your actual second-level data API call.
    """
    # Create a time series index for second-level data
    time_index = pd.date_range(start_date, end_date, freq='S') # frequency by second
    num_points = len(time_index)
    data = {
        'Open':   pd.Series(np.random.uniform(100, 150, num_points), index=time_index),
        'Close':  pd.Series(np.random.uniform(100, 150, num_points), index=time_index),
        'High':   pd.Series(np.random.uniform(100, 150, num_points), index=time_index),
        'Low':    pd.Series(np.random.uniform(100, 150, num_points), index=time_index),
    }
    df = pd.DataFrame(data)
    return df
