# modules/logic_soft.py

import pandas as pd
import numpy as np
import optuna
import logging
from typing import Dict, Any

from modules.indicators import calculate_sma, calculate_rsi, calculate_adx, calculate_macd, calculate_atr

logger = logging.getLogger("OptunaApp")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def logit(p):
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return np.log(p / (1 - p))

def tanh(x):
    return np.tanh(x)

def bullish_sma(sma_short, sma_long):
    if sma_long == 0:
        return 0.0
    return sigmoid(sma_short / sma_long)

def bullish_rsi(rsi, rsi_threshold):
    if rsi == 0 or rsi > 100:
        return 0.0
    if rsi < rsi_threshold:
        return sigmoid((rsi_threshold - rsi) / rsi_threshold)
    return 0.0

def bullish_adx(adx, adx_threshold):
    if adx == 0:
        return 0.0
    if adx > 100:
        return 1.0
    if adx > adx_threshold:
        return sigmoid((adx - adx_threshold) / (100 - adx_threshold))
    return 0.0

def bullish_macd(macd_line, signal_line):
    if macd_line == 0:
        return 0.0
    return sigmoid(macd_line - signal_line)

def generate_soft_signal(data: pd.DataFrame, weights: Dict[str, float], rsi_threshold: float, adx_threshold: float) -> pd.Series:
    sma_short = data['SMA_Short']
    sma_long = data['SMA_Long']
    rsi = data['RSI']
    adx = data['ADX']
    macd_line = data['MACD_Line']
    signal_line = data['Signal_Line']

    soft_signals = []

    for i in range(len(data)):
        score = (
            weights["sma_weight"] * logit(bullish_sma(sma_short.iloc[i], sma_long.iloc[i])) +
            weights["rsi_weight"] * logit(bullish_rsi(rsi.iloc[i], rsi_threshold)) +
            weights["adx_weight"] * logit(bullish_adx(adx.iloc[i], adx_threshold)) +
            weights["macd_weight"] * logit(bullish_macd(macd_line.iloc[i], signal_line.iloc[i]))
        )
        soft_signals.append(tanh(score))

    return pd.Series(soft_signals, index=data.index)

def backtest_with_soft_decisions(
    data: pd.DataFrame,
    params: Dict[str,Any],
    trial: optuna.trial.Trial=None,
    detailed_logs: bool = False
):

    short_window = params.get('short_window', 20)
    long_window = params.get('long_window', 100)
    rsi_threshold = params.get('rsi_threshold', 30)
    adx_threshold = params.get('adx_threshold', 25)
    atr_multiplier = params.get('atr_multiplier', 1.5)
    transaction_cost = params.get('transaction_cost', 0.001)
    stop_loss = params.get('stop_loss', 0.02)
    take_profit = params.get('take_profit', 0.05)
    slippage = params.get('slippage', 0.001)

    weights = {
        "sma_weight": params.get("sma_weight", 1.0),
        "rsi_weight": params.get("rsi_weight", 1.0),
        "adx_weight": params.get("adx_weight", 1.0),
        "macd_weight": params.get("macd_weight", 1.0)
    }

    data_copy = data.copy()
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
            logger.info("No data after dropna => returning Sharpe=0 (soft logic)")
        return 0, {}

    data_copy['Soft_Signal'] = generate_soft_signal(data_copy, weights, rsi_threshold, adx_threshold)

    # naive approach: if Soft_Signal > 0 => buy, else 0 => no position
    data_copy['Position'] = np.where(data_copy['Soft_Signal'] > 0, 1, 0)
    data_copy['Position'] = data_copy['Position'].ffill().fillna(0)

    data_copy['Entry_Price'] = data_copy['Open'].shift(1).where(data_copy['Position'] > 0, np.nan)
    data_copy['Entry_Price'] = data_copy['Entry_Price'].ffill()
    data_copy['Price_Change'] = (data_copy['Close'] - data_copy['Entry_Price']) / data_copy['Entry_Price']

    data_copy['Stop_Loss'] = -atr_multiplier*data_copy['ATR']
    data_copy['Take_Profit'] = atr_multiplier*data_copy['ATR']

    data_copy['Position'] = np.where(
        (data_copy['Position'] > 0) & (data_copy['Price_Change'] <= data_copy['Stop_Loss']),
        0,
        data_copy['Position']
    )
    data_copy['Position'] = np.where(
        (data_copy['Position'] > 0) & (data_copy['Price_Change'] >= take_profit),
        0,
        data_copy['Position']
    )

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
        logger.info(f"Trial #{trial.number} partial(soft) => Sharpe={sharpe:.4f}, params={params}")
    return sharpe, {}

def objective_with_soft_logic(
    trial: optuna.trial.Trial,
    data: pd.DataFrame,
    cross_val_folds: int=1,
    detailed_logs: bool=False
) -> float:
    """
    This objective function uses the 'soft/LLR' approach instead of the normal logic.
    """
    short_window = trial.suggest_int('short_window', 12, 22, step=1)
    long_window = trial.suggest_int('long_window', 90, 120, step=5)
    rsi_threshold = trial.suggest_float('rsi_threshold', 18, 28, step=1)
    adx_threshold = trial.suggest_float('adx_threshold', 18, 32, step=1)
    atr_multiplier = trial.suggest_float('atr_multiplier', 1.1, 1.4, step=0.1)
    stop_loss = trial.suggest_float('stop_loss', 0.015, 0.02, step=0.001)
    take_profit = trial.suggest_float('take_profit', 0.045, 0.055, step=0.001)
    transaction_cost = trial.suggest_float('transaction_cost', 0.0005, 0.001, step=0.0005)
    slippage = trial.suggest_float('slippage', 0.0005, 0.001, step=0.0005)

    sma_weight = trial.suggest_float('sma_weight', 0.0, 1.0)
    rsi_weight = trial.suggest_float('rsi_weight', 0.0, 1.0)
    adx_weight = trial.suggest_float('adx_weight', 0.0, 1.0)
    macd_weight = trial.suggest_float('macd_weight', 0.0, 1.0)

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
        'sma_weight': sma_weight,
        'rsi_weight': rsi_weight,
        'adx_weight': adx_weight,
        'macd_weight': macd_weight
    }

    # We do not do multi-fold CV here, but you can if you want. This is single-chunk for simplicity.
    # If you needed cross_val, you'd do a similar loop as in backtest_sma_crossover_adx.

    sharpe, _ = backtest_with_soft_decisions(data, params, trial, detailed_logs)
    logger.info(f"Trial #{trial.number} Soft Results: Sharpe={sharpe:.5f}, Params={params}")
    return sharpe

def run_trial_fn_factory_soft(
    data_f: pd.DataFrame,
    cross_val_folds: int,
    detailed_logs: bool
):
    """
    We create a closure for the soft logic approach.
    """
    def run_trial_fn(trial: optuna.trial.Trial) -> float:
        return objective_with_soft_logic(trial, data_f, cross_val_folds, detailed_logs)
    return run_trial_fn
