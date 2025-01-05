# modules/noise.py
import numpy as np
import pandas as pd
from typing import Literal

def add_white_noise(data: pd.DataFrame, noise_intensity: float, column_name:str = "Close") -> pd.DataFrame:
    """ Adds White Noise to a specific column.

        Args:
          data (pd.DataFrame): input dataframe
          noise_intensity (float)  :  std value to generate a new random noise component via numpy's standard normal distribution
          column_name(str): the specific columns where the data goes into with new noise
    Returns:
        pd.DataFrame: noisy signal with added 'noisy' reference to the specified col name such as Close_noisy column.
    Raises:
        KeyError: this usually means that column does not exist, ensure parameters are setup!
        """

    noise = np.random.normal(0, noise_intensity, len(data))
    data[f'{column_name}_Noisy'] = data[column_name] + noise
    return data

def add_brownian_noise(data:pd.DataFrame, noise_intensity: float, column_name:str = "Close") -> pd.DataFrame:
    """ Brownian / Random Walk. This generates a random noise which moves the close forward.

    Args:
        data(pd.DataFrame)      :input data for brownian process (cumulative)
        noise_intensity(float)     : the scale for this diffusion
         column_name(str)  : target the close data column.
      Returns:
        pd.DataFrame   : A new data column added via .cumsum to move signal around the data specified
      Raises:
           KeyError: If there is an issue fetching that name. Check your naming parameters for consistency
        """
    noise = np.random.normal(0, noise_intensity, len(data))
    brownian_motion = np.cumsum(noise)
    data[f'{column_name}_Noisy'] = data[column_name] + brownian_motion
    return data


def add_autoregressive_noise(data: pd.DataFrame, noise_intensity: float, ar_coeff:float=0.5, column_name:str = "Close") -> pd.DataFrame:
    """ Add autoregressive signal to a single column
      Args:
            data(pd.DataFrame)      :input dataframe with close prices
            noise_intensity (float)  : how much to inject noise (scale).
            ar_coeff (float) : coefficent that uses the prior iteration to inject, should always be from [0..1.0] normally
            column_name(str)  : to indicate what column that goes into as noise + col name = result column output
    Returns:
        pd.DataFrame  :  new col appended for new 'Noisy signal' by previous plus small random step process.
        Raises:
          KeyError: Data column incorrect?

       """

    noise = [np.random.normal(0, noise_intensity)] # use an init single element to generate series correctly via a seed
    for _ in range(1, len(data)):
        noise.append(ar_coeff * noise[-1] + np.random.normal(0, noise_intensity))
    data[f'{column_name}_Noisy'] = data[column_name] + noise
    return data

def add_cyclic_noise(data: pd.DataFrame, noise_intensity: float, frequency: float = 0.1, column_name:str = "Close") -> pd.DataFrame:
    """ Adds a simple Sinusoid / cyclic pattern
    Args:
         data (pd.DataFrame): price data.
         noise_intensity(float)  : scale the output to create large or small movements via an amplitude (intensity here).
         frequency(float)       : how fast in cycles will this happen
         column_name(str)       :  add suffix for this specific column to track output.
    Returns:
        pd.DataFrame   : Data with sinusoid appended using numpy sine fn, noise plus column,
    Raises:
        KeyError: invalid column naming or formatting in params
        """

    t = np.arange(len(data))
    noise = noise_intensity * np.sin(2 * np.pi * frequency * t)
    data[f'{column_name}_Noisy'] = data[column_name] + noise
    return data

def distort_signal(data: pd.DataFrame, distort_intensity: float, column_name:str = "Close", distrot_type:Literal["moving_average"] = "moving_average")-> pd.DataFrame:
    """Distorts a signal
        Args:
           data (pd.DataFrame)     : price series with close information etc, open low, high too.
           distort_intensity (float) :  used as weight scale
           column_name(str)          : what data the distrot signal should be appended on top with suffix (_Distorted)
           distrot_type(str)         : type of distort we use "moving_average". More can be made such as via FFT transform or simply gaussian filter etc.
         Returns:
            pd.DataFrame  with new column containing a signal from the `distort_type`.
          Raises:
              KeyError: ensures valid `column_name` to inject data!
        """

    if distrot_type == "moving_average":
        data_copy = data.copy() # to avoid side effects (return a value)!
        data_copy['SMA_Short'] = data_copy[column_name].rolling(window=5, min_periods=1).mean()
        data[f'{column_name}_Distorted'] = data[column_name] + distort_intensity * (data_copy['SMA_Short'] - data_copy[column_name])
    return data
