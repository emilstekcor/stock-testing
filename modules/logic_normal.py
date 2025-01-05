# modules/logic_normal.py

import pandas as pd
import numpy as np
import logging
import optuna
from typing import Dict, Any

from modules.indicators import (
    calculate_sma, calculate_rsi, calculate_adx, calculate_macd, calculate_atr
)
from modules.transforms import apply_causal_filter, apply_fft, calculate_volatility_indicator, apply_stl
from modules.noise import add_white_noise, add_brownian_noise, add_autoregressive_noise, add_cyclic_noise, distort_signal
from modules.trades_logic import apply_stop_loss_take_profit

logger = logging.getLogger("OptunaApp")


def _backtest_sma_crossover_adx_single(
    data: pd.DataFrame,
    params: Dict[str,Any],
    trial: optuna.trial.Trial = None,
    detailed_logs: bool = False
) -> (float, Dict[str,Any]):
    
    short_window = params.get('short_window', 20)
    long_window = params.get('long_window', 100)
    rsi_threshold = params.get('rsi_threshold', 30)
    adx_threshold = params.get('adx_threshold', 25)
    transaction_cost = params.get('transaction_cost', 0.001)
    slippage = params.get('slippage', 0.001)


    data_copy = data.copy()
    try:
        data_copy['SMA_Short'] = calculate_sma(data_copy, short_window)
        data_copy['SMA_Long'] = calculate_sma(data_copy, long_window)
        data_copy['RSI'] = calculate_rsi(data_copy)
        data_copy['ATR'] = calculate_atr(data_copy)
        data_copy['ADX'] = calculate_adx(data_copy)
        macd = calculate_macd(data_copy)
        data_copy = pd.concat([data_copy, macd], axis=1)
        
        data_copy.dropna(subset=['SMA_Short','SMA_Long','RSI','ATR','ADX','MACD_Line','Signal_Line'], inplace=True)
        if data_copy.empty:
            if detailed_logs:
                logger.info("No data left after dropna => returning Sharpe=0")
            return 0, {}

        data_copy['Signal'] = np.where(
            (data_copy['SMA_Short'] > data_copy['SMA_Long']) &
            (data_copy['RSI'] < rsi_threshold) &
            (data_copy['ADX'] > adx_threshold) &
            (data_copy['MACD_Line'] > data_copy['Signal_Line']),
            1,
            np.where(
                (data_copy['SMA_Short'] < data_copy['SMA_Long']) |
                (data_copy['RSI'] > 70) |
                (data_copy['ADX'] < adx_threshold) |
                (data_copy['MACD_Line'] < data_copy['Signal_Line']),
                -1, 0
            )
        )
        data_copy['Signal'] = data_copy['Signal'].diff()
        data_copy['Position'] = 0
        data_copy['Position'] = np.where(data_copy['Signal'] == 1, 1,
                                        np.where(data_copy['Signal'] == -1, -1, 0))
        data_copy['Position'] = data_copy['Position'].ffill().fillna(0)
        data_copy = apply_stop_loss_take_profit(data_copy, params)

        data_copy['Signal_RM'] = data_copy['Position'].diff().fillna(0)

        data_copy['Strategy_Return'] = data_copy['Position'].shift(1)*(data_copy['Close'].pct_change() - slippage)
        data_copy['Strategy_Return'] = data_copy['Strategy_Return'].fillna(0)

        data_copy['Trades'] = data_copy['Signal_RM'].abs()
        data_copy['Strategy_Return'] -= (transaction_cost * data_copy['Trades'])

        data_copy['Cumulative_Strategy'] = (1 + data_copy['Strategy_Return']).cumprod()
        std_ret = data_copy['Strategy_Return'].std()

        if std_ret == 0:
            sharpe = 0
        else:
            sharpe = (data_copy['Strategy_Return'].mean() / std_ret) * np.sqrt(252)

        if detailed_logs and trial is not None:
            logger.info(f"Trial #{trial.number} partial => Sharpe {sharpe:.5f}, Params={params}")

        return sharpe, {}
    except Exception as e:
        if detailed_logs:
            logger.warning(f"Exception in backtest: {e}, returning Sharpe=0")
        return 0, {}

def backtest_sma_crossover_adx(
    data: pd.DataFrame,
    params: Dict[str,Any],
    cross_val_folds: int = 1,
    trial: optuna.trial.Trial = None,
    detailed_logs: bool = False
) -> (float, Dict[str,Any]):

    if cross_val_folds <= 1:
        return _backtest_sma_crossover_adx_single(data, params, trial, detailed_logs)
    else:
        folds = np.array_split(data, cross_val_folds)
        sharpe_scores = []
        for fold_index, fold_df in enumerate(folds, start=1):
            s, _ = _backtest_sma_crossover_adx_single(fold_df, params, trial, detailed_logs)
            sharpe_scores.append(s)
            if trial is not None:
                trial.report(np.mean(sharpe_scores), step=fold_index)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
        mean_sharpe = np.mean(sharpe_scores)
        return mean_sharpe, {"MeanCVSharpe": mean_sharpe}

# Extreme short strategy parameters
short_window_extreme = 5
long_window_extreme = 20
atr_multiplier_extreme = 1.0
stop_loss_ticks_extreme = 3
profit_ticks_extreme = 5


def extreme_short_trading_logic(data: pd.DataFrame, params:Dict[str,Any]) -> pd.DataFrame:
    """
    Implements a scalping strategy for extreme short trading using RSI and ATR.
    Parameters:
        data (pd.DataFrame): DataFrame with OHLC data.
        params (Dict[str,Any]): dictionary with the parameters to run the backtest

    Returns:
        pd.DataFrame: DataFrame with added signals and position columns.
    """
    data = data.copy()

    # Calculate Indicators
    data['RSI'] = calculate_rsi(data, params.get('short_window_extreme', short_window_extreme))
    data['ATR'] = calculate_atr(data, params.get('short_window_extreme', short_window_extreme))
    
    # Generate Trading Signals
    data['Signal'] = 0
    data.loc[(data['RSI'] < params.get('rsi_threshold_low', 30)), 'Signal'] = 1  # Buy signal
    data.loc[(data['RSI'] > params.get('rsi_threshold_high', 70)), 'Signal'] = -1  # Sell signal

    # Positions based on signals
    data['Position'] = data['Signal'].ffill().fillna(0)
    
    
    return data
def objective(
    trial: optuna.trial.Trial,
    data: pd.DataFrame,
    cross_val_folds: int=1,
    detailed_logs: bool=False,
    feature_selection: Dict[str, bool] = None,
    noise_params: Dict[str, Any] = None,
    extreme_short_params:Dict[str,bool] = None
) -> float:
    short_window = trial.suggest_int('short_window', 12, 22, step=1)
    long_window = trial.suggest_int('long_window', 90, 120, step=5)
    rsi_threshold = trial.suggest_float('rsi_threshold', 18, 28, step=1)
    adx_threshold = trial.suggest_float('adx_threshold', 18, 32, step=1)
    atr_multiplier = trial.suggest_float('atr_multiplier', 1.1, 1.4, step=0.1)
    stop_loss = trial.suggest_float('stop_loss', 0.015, 0.02, step=0.001)
    take_profit = trial.suggest_float('take_profit', 0.045, 0.055, step=0.001)
    transaction_cost = trial.suggest_float('transaction_cost', 0.0005, 0.001, step=0.0005)
    slippage = trial.suggest_float('slippage', 0.0005, 0.001, step=0.0005)
    
    short_window_extreme_param = trial.suggest_int('short_window_extreme', 2, 10, step=1)
    rsi_threshold_low_param = trial.suggest_float('rsi_threshold_low', 20, 40, step=1)
    rsi_threshold_high_param = trial.suggest_float('rsi_threshold_high', 60, 80, step=1)
    atr_multiplier_extreme_param = trial.suggest_float('atr_multiplier_extreme', 0.5, 1.5, step=0.1)
    stop_loss_ticks_extreme_param = trial.suggest_int('stop_loss_ticks_extreme', 1, 5, step=1)
    profit_ticks_extreme_param = trial.suggest_int('profit_ticks_extreme', 3, 10, step=1)
    

    params = {
            'short_window': short_window,
            'long_window': long_window,
            'rsi_threshold': rsi_threshold,
            'adx_threshold': adx_threshold,
            'atr_multiplier': atr_multiplier,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'transaction_cost': transaction_cost,
            'slippage': slippage,
             'short_window_extreme': short_window_extreme_param,
            'rsi_threshold_low': rsi_threshold_low_param,
            'rsi_threshold_high': rsi_threshold_high_param,
            'atr_multiplier_extreme': atr_multiplier_extreme_param,
            'stop_loss_ticks_extreme': stop_loss_ticks_extreme_param,
            'profit_ticks_extreme': profit_ticks_extreme_param
        }

    filter_enabled = feature_selection.get('filter_enabled', False) if feature_selection else False
    fft_enabled = feature_selection.get('fft_enabled', False) if feature_selection else False
    volatility_enabled = feature_selection.get('volatility_enabled', False) if feature_selection else False
    stl_enabled = feature_selection.get('stl_enabled', False) if feature_selection else False
    use_noise = noise_params.get("use_noise", False) if noise_params else False
    use_extreme_short = extreme_short_params.get("use_extreme_short", False) if extreme_short_params else False
    noise_type = trial.suggest_categorical("noise_type", ["white", "brownian", "autoregressive", "cyclic", "distort_ma"]) if use_noise else None
    noise_intensity = trial.suggest_float("noise_intensity", 0.0, 0.1) if use_noise else 0.0
    ar_coeff = trial.suggest_float("ar_coeff", 0.0, 1.0) if use_noise and noise_type == "autoregressive" else 0.5
    cyclic_frequency = trial.suggest_float("cyclic_frequency", 0.01, 0.2) if use_noise and noise_type == "cyclic" else 0.1
    distort_intensity = trial.suggest_float("distort_intensity", 0.0, 0.1) if use_noise and noise_type == "distort_ma" else 0.0
    
    filter_params = {
        'filter_order': trial.suggest_int('filter_order', 2, 5, step=1) if filter_enabled else 3,
        'filter_cutoff': trial.suggest_float('filter_cutoff', 0.05, 0.3, step=0.05) if filter_enabled else 0.1,
    }
    fft_params = {
        'fft_window_size': trial.suggest_int('fft_window_size', 10, 60, step=10) if fft_enabled else 30
    }
    volatility_params = {
        'volatility_window': trial.suggest_int("volatility_window", 10, 40, step=10) if volatility_enabled else 20
    }
    stl_params = {
        'seasonal_period': trial.suggest_int('stl_seasonal_period', 7, 30, step=7) if stl_enabled else 21,
    }


    data_copy = data.copy()
    if filter_enabled:
        data_copy = apply_causal_filter(data_copy, filter_params)
        if 'Filtered_Close' in data_copy:
            data_copy['Close'] = data_copy['Filtered_Close']
    if fft_enabled:
        data_copy = apply_fft(data_copy, fft_params)
    if volatility_enabled:
        data_copy = calculate_volatility_indicator(data_copy, volatility_params)
    if stl_enabled:
      data_copy = apply_stl(data_copy, stl_params)

    if use_noise:
        if noise_type == "white":
            data_copy = add_white_noise(data_copy, noise_intensity)
        elif noise_type == "brownian":
             data_copy = add_brownian_noise(data_copy, noise_intensity)
        elif noise_type == "autoregressive":
             data_copy = add_autoregressive_noise(data_copy, noise_intensity, ar_coeff)
        elif noise_type == "cyclic":
             data_copy = add_cyclic_noise(data_copy, noise_intensity, cyclic_frequency)
        elif noise_type == "distort_ma":
             data_copy = distort_signal(data_copy, distort_intensity)
        if 'Close_Noisy' in data_copy:
            data_copy['Close'] = data_copy['Close_Noisy']
        if 'Close_Distorted' in data_copy:
            data_copy['Close'] = data_copy['Close_Distorted']
    if use_extreme_short:
        data_copy = extreme_short_trading_logic(data_copy, params)

    
    sharpe, metrics = backtest_sma_crossover_adx(data_copy, params, cross_val_folds, trial, detailed_logs) if not use_extreme_short else backtest_sma_crossover_adx(data_copy, params, cross_val_folds, trial, detailed_logs)
    logger.info(f"Trial #{trial.number} Results: Sharpe={sharpe:.5f}, Params={params}")
    return sharpe

def run_trial_fn_factory(
    data_f: pd.DataFrame,
    cross_val_folds: int,
    detailed_logs: bool
):
    """
    We create a closure for the normal logic approach.
    """
    def run_trial_fn(trial: optuna.trial.Trial) -> float:
         feature_selection = trial.user_attrs.get('feature_selection', {})
         noise_params = trial.user_attrs.get('noise_params', {})
         extreme_short_params = trial.user_attrs.get('extreme_short_params', {})
         return objective(trial, data_f, cross_val_folds, detailed_logs, feature_selection, noise_params, extreme_short_params)
    return run_trial_fn
