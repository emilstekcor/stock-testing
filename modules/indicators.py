# modules/indicators.py
import pandas as pd
import numpy as np
from typing import List

def calculate_sma(data: pd.DataFrame, window: int, column_name: str = 'Close') -> pd.Series:
    """Calculates the Simple Moving Average (SMA) for a given column.
        Args:
            data (pd.DataFrame): data that is used to generate the indicator
            window (int)       : window size to roll
            column_name (str)   : name of price to use such as "Open", or "Close"
        Returns:
             pd.Series:   new series indicator, it can be used for charting and data transformations.
        Raises:
            KeyError: If there column_name not exist, ensure data is formatted
        """

    return data[column_name].rolling(window=window, min_periods=1).mean()

def calculate_rsi(data: pd.DataFrame, window: int = 14, column_name: str = 'Close') -> pd.Series:
    """Calculates the Relative Strength Index (RSI) for given series.
        Args:
            data (pd.DataFrame): data that is used to generate the indicator
            window (int)        : size for calculating change / momentum
            column_name (str) :  name of price series to use such as "Open" or "Close"
        Returns:
           pd.Series: series containing the RSI values.
         Raises:
             KeyError: if the data column provided in col_name is not correct for the DataFrame
            """

    delta = data[column_name].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs =  avg_gain / avg_loss # protect this div zero
    rsi = 100 - (100 / (1 + rs))
    rsi.fillna(0, inplace = True)
    return rsi
def calculate_macd(data: pd.DataFrame, span_short:int=12, span_long:int=26, span_signal:int=9, column_name:str = 'Close') -> pd.DataFrame:
    """Calculates the MACD line and Signal line.
       Args:
            data (pd.DataFrame): data to extract macd/signal line indicators from.
            span_short (int)        : fast rolling window period
            span_long (int) : slow rolling window period. Must be greater than `span_short`.
            span_signal(int)   : how long for exponential mov avg
            column_name (str)   : which price series you use as the MACD input from `data`
        Returns:
            pd.DataFrame with 'MACD_Line' and 'Signal_Line'.
         Raises:
            KeyError:  when you put column name for `data` but no column by such name exist for processing

    """

    ema_short = data[column_name].ewm(span=span_short, adjust=False).mean()
    ema_long = data[column_name].ewm(span=span_long, adjust=False).mean()
    macd_line = ema_short - ema_long
    signal_line = macd_line.ewm(span=span_signal, adjust=False).mean()
    return pd.DataFrame({'MACD_Line': macd_line, 'Signal_Line': signal_line})

def calculate_atr(data: pd.DataFrame, window: int = 14, high_column: str = 'High', low_column: str = 'Low', close_column: str ='Close') -> pd.Series:
    """Calculates the Average True Range (ATR).
        Args:
            data (pd.DataFrame):  to generate atr based on high, low and close price values
            window (int)       :  size to use for rolling mean
            high_column(str)  : column to use as high values for atr processing
            low_column (str)  : column to use for the lows, this is combined
            close_column (str) : which is the correct close data point
        Returns:
           pd.Series: the atr over price series.
         Raises:
            KeyError: check if `data` is correctly named if error.
        """

    high_low = data[high_column] - data[low_column]
    high_close = np.abs(data[high_column] - data[close_column].shift())
    low_close = np.abs(data[low_column] - data[close_column].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=window, min_periods=1).mean()

def calculate_adx(data: pd.DataFrame, window: int = 14, high_column: str = 'High', low_column: str = 'Low', close_column:str = 'Close') -> pd.Series:
     """Calculates the Average Directional Index (ADX).
        Args:
           data (pd.DataFrame)       : data for the ADX, using high, low and close
           window (int)      : rolling size for how far to average data.
           high_column(str)       : string reference in `data` column as Highs, not None.
           low_column (str)        :   low columns ref as str in `data` for caluclating the difference between. not None.
            close_column(str): str close data used to track range between lows and highs to estimate power for movement
        Returns:
           pd.Series:  The adx indicator series generated for backtesting/
         Raises:
             KeyError: thrown when dataframe do not exist the named cols.
            """

     plus_dm = data[high_column].diff().where(data[high_column].diff() > data[low_column].diff(), 0)
     minus_dm = data[low_column].diff().where(data[low_column].diff() > data[high_column].diff(), 0)
     plus_dm_smoothed = plus_dm.rolling(window=window, min_periods=1).mean()
     minus_dm_smoothed = minus_dm.rolling(window=window, min_periods=1).mean()
     atr = calculate_atr(data, window, high_column, low_column, close_column) # note same config goes down here

     plus_di = (plus_dm_smoothed / atr) * 100
     minus_di = (minus_dm_smoothed / atr) * 100
     dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100 # if div 0 is the only exception this should always work


     return dx.rolling(window=window, min_periods=1).mean()
