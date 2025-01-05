import pandas as pd
import numpy as np
from scipy import signal
from statsmodels.tsa.seasonal import STL
import logging

logger = logging.getLogger(__name__)

def apply_causal_filter(data: pd.DataFrame, filter_params: dict) -> pd.DataFrame:
    """
    Applies a zero-phase low-pass filter (via filtfilt) to the 'Close' column of a DataFrame.

    NOTE: 'filtfilt' is an acausal filter. If you truly want a causal filter, 
    use 'signal.lfilter' or rename the function accordingly.

    Parameters:
    ----------
    data : pd.DataFrame
        DataFrame containing at least a 'Close' column.
    filter_params : dict
        Dictionary that may include:
         - 'filter_order' (int): Filter order. Default is 3.
         - 'filter_cutoff' (float): Normalized cutoff frequency (0 < cutoff < 1). Default is 0.1.

    Returns:
    -------
    pd.DataFrame
        The same DataFrame with a new column 'Filtered_Close'.
    """
    if data.empty:
        logger.warning("Data is empty. No filter applied.")
        return data

    order = filter_params.get('filter_order', 3)
    cutoff = filter_params.get('filter_cutoff', 0.1)

    try:
        b, a = signal.butter(order, cutoff, btype='low', analog=False)
        filtered_close = signal.filtfilt(b, a, data['Close'])
        data['Filtered_Close'] = filtered_close
    except Exception as e:
        logger.error(f"Error applying filtfilt: {e}")
    
    return data

def apply_fft(data: pd.DataFrame, fft_params: dict) -> pd.DataFrame:
    """
    Computes the FFT of the 'Close' column and stores the magnitude in 'FFT_Abs'.

    Parameters:
    ----------
    data : pd.DataFrame
        DataFrame containing at least a 'Close' column.
    fft_params : dict
        Dictionary that may include:
         - 'fft_window_size' (int): Not currently used for partial-FFT, 
           but could be used if you plan to apply a window or segment the data.

    Returns:
    -------
    pd.DataFrame
        The DataFrame with an added 'FFT_Abs' column.
    """
    if data.empty:
        logger.warning("Data is empty. No FFT applied.")
        return data

    window_size = fft_params.get('fft_window_size', 30)
    close = data['Close'].values
    if len(close) < window_size:
        logger.info(f"Data length < window_size ({window_size}). Skipping FFT.")
        return data
    
    try:
        fft_result = np.fft.fft(close)
        fft_abs = np.abs(fft_result)
        data['FFT_Abs'] = pd.Series(fft_abs, index=data.index)
    except Exception as e:
        logger.error(f"Error applying FFT: {e}")

    return data

def calculate_volatility_indicator(data: pd.DataFrame, volatility_params: dict) -> pd.DataFrame:
    """
    Calculates rolling volatility (standard deviation) of the 'Close' column.

    Parameters:
    ----------
    data : pd.DataFrame
        DataFrame containing at least a 'Close' column.
    volatility_params : dict
        Dictionary that may include:
         - 'volatility_window' (int): Rolling window size for volatility. Default is 20.

    Returns:
    -------
    pd.DataFrame
        DataFrame with a new column 'Rolling_Volatility'.
    """
    if data.empty:
        logger.warning("Data is empty. No volatility computed.")
        return data

    window = volatility_params.get("volatility_window", 20)
    try:
        data['Rolling_Volatility'] = data['Close'].rolling(window=window, min_periods=1).std()
    except Exception as e:
        logger.error(f"Error calculating rolling volatility: {e}")

    return data

def apply_stl(data: pd.DataFrame, stl_params: dict) -> pd.DataFrame:
    """
    Applies STL decomposition to the 'Close' column, adding columns for trend, seasonal, and residual.

    Parameters:
    ----------
    data : pd.DataFrame
        DataFrame containing at least a 'Close' column.
    stl_params : dict
        Dictionary that may include:
         - 'seasonal_period' (int): The seasonal period (e.g., 21 for roughly monthly in daily data).

    Returns:
    -------
    pd.DataFrame
        DataFrame with added columns 'STL_Trend', 'STL_Seasonal', 'STL_Residual'.
    """
    if data.empty:
        logger.warning("Data is empty. No STL decomposition applied.")
        return data

    seasonal_period = stl_params.get('seasonal_period', 21)
    if len(data) < seasonal_period:
        logger.info(f"Data length < seasonal_period ({seasonal_period}). Skipping STL.")
        return data
    
    try:
        stl = STL(data['Close'], seasonal=seasonal_period)
        result = stl.fit()
        data['STL_Trend'] = result.trend
        data['STL_Seasonal'] = result.seasonal
        data['STL_Residual'] = result.resid
    except Exception as e:
        logger.warning(f"STL decomposition failed: {e}")

    return data
