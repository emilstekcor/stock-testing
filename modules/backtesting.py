import pandas as pd # <--- ADDED THIS LINE
import numpy as np
import logging
from typing import Dict, Any
from modules.trades_logic import apply_stop_loss_take_profit
from modules.indicators import calculate_sma, calculate_rsi, calculate_adx, calculate_macd, calculate_atr
from modules.trades_on_price import plot_trades_on_price
from modules.equity_curve import calculate_max_drawdown, calculate_sortino_ratio
logger = logging.getLogger(__name__)

def backtest_with_metrics(
    data: pd.DataFrame,
    params: Dict[str, Any],
    detailed_logs: bool = False,
    soft_logic: bool = False,
    extreme_short: bool = False,
    price_column:str = 'Close',
    entry_price_type:str = "Open"
) -> Dict[str, Any]:
    """
     Main backtesting function, with options of choosing soft or hard/ normal/ extreme approach for the positions!. Dataframe needs not any other parameters than a position parameter - using trading simulation logic module for any trading system.
      Must also apply with stop losses (using price + type from backtesting / module setting parameter or data default value). Returns the metrics used in simulator
    """
    sharpe = 0 # init those as zero for default.
    std_ret = 0 # default for type check!
    try:
        data = data.copy()
        if soft_logic:
             # Run soft logic: with correct signal + position -> then apply trades/stop / logic and also trade simulation rules to ensure same system for calculating trade entry + exits! using configuration as parameter that is correct always since this also gets settings that you apply using single logical point with specific rules. Everything!. So trading simulator correctly applies at each part
           data = _calculate_indicators(data, params, price_column)
           data.dropna(subset=['SMA_Short','SMA_Long','RSI','ATR','ADX','MACD_Line','Signal_Line'], inplace=True)
           if data.empty:
               return {"error": "No data after preprocessing",
                       "sharpe":0,
                       "max_drawdown": 0,
                       "sortino": 0,
                        "trades": 0,
                        'trades_details':pd.DataFrame(),
                       "equity_curve": pd.Series(),
                       "trades_on_price": None
                    }
           _generate_soft_signals(data,params)
           apply_stop_loss_take_profit(data, params,price_column, entry_price_type)
           data, trade_df =  _calculate_trades(data,params, price_column)
           equity_curve = (1 + data['Strategy_Return']).cumprod()
        elif extreme_short:
           data = extreme_short_trading_logic(data, params, price_column)
           apply_stop_loss_take_profit(data,params,price_column, entry_price_type)
           data, trade_df  = _calculate_trades(data,params, price_column)
           equity_curve = (1 + data['Strategy_Return']).cumprod()
           std_ret = data['Strategy_Return'].std()
           if std_ret != 0:
                sharpe = (data['Strategy_Return'].mean() / std_ret) * np.sqrt(252)
        else:
            data = _calculate_indicators(data, params, price_column)
            data.dropna(subset=['SMA_Short','SMA_Long','RSI','ATR','ADX','MACD_Line','Signal_Line'], inplace=True)
            if data.empty:
               return {"error": "No data after preprocessing",
                       "sharpe":0,
                       "max_drawdown": 0,
                       "sortino": 0,
                       "trades": 0,
                       'trades_details':pd.DataFrame(),
                        "equity_curve": pd.Series(),
                       "trades_on_price": None
                    }
            _generate_normal_signals(data,params)
            apply_stop_loss_take_profit(data,params, price_column, entry_price_type)
            data, trade_df = _calculate_trades(data,params, price_column)
            equity_curve = (1 + data['Strategy_Return']).cumprod()
        trades_on_price = plot_trades_on_price(data, data.index.name, price_column)
        max_drawdown = calculate_max_drawdown(equity_curve)
        returns = data['Strategy_Return']
        sortino = calculate_sortino_ratio(returns)
        if not extreme_short:
             std_ret = data['Strategy_Return'].std()
             if std_ret != 0:
                  sharpe = (data['Strategy_Return'].mean() / std_ret) * np.sqrt(252)
        return {
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "sortino": sortino,
            "trades": data['Trades'].iloc[0] if not data.empty else 0,
            "trades_details": trade_df,
            "equity_curve": equity_curve,
            "trades_on_price": trades_on_price
        }
    except Exception as e:
        logger.error(f"Error in backtesting: {e}", exc_info=True)
        return {
               "error": f"Error in backtesting: {e}",
                "sharpe":0,
                "max_drawdown": 0,
                "sortino": 0,
               "trades": 0,
               "trades_details":pd.DataFrame(),
                "equity_curve": pd.Series(),
                "trades_on_price": None
            }
