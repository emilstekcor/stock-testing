# modules/trades_logic.py

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def apply_stop_loss_take_profit(
    data: pd.DataFrame,
    stop_loss: float = 0.05,
    take_profit: float = 0.10,
    price_column: str = 'Close',
    position_column: str = 'Position'
) -> pd.DataFrame:
    """
    Applies stop-loss and take-profit logic to the 'data' DataFrame.

    This example assumes:
      - 'Position' column indicates when you are in a trade (1 for long, 0 for flat).
      - 'EntryPrice' column stores the price at which the trade was opened.
      - 'Close' column (by default) stores the latest price used for checking stops/profits.
      - 'stop_loss' and 'take_profit' are fractional thresholds from the entry price.

    The function closes the trade (sets Position back to 0) if the stop-loss
    or take-profit levels are hit. If you need more nuanced logic (e.g., partial exits),
    you can customize accordingly.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame that must include at least 'Close', 'Position', and 'EntryPrice' columns
        for this logic to work as is. If you have a different structure, adapt the code.
    stop_loss : float, optional
        Percentage-based stop-loss threshold. Default is 0.05 (i.e., 5% below entry).
    take_profit : float, optional
        Percentage-based take-profit threshold. Default is 0.10 (i.e., 10% above entry).
    price_column : str, optional
        Name of the column used to check current price (default 'Close').
    position_column : str, optional
        Name of the column that indicates whether a trade is active (default 'Position').

    Returns
    -------
    pd.DataFrame
        Modified DataFrame with updated 'Position' where SL/TP triggers are met.

    Example
    -------
    >>> df = pd.DataFrame({
    ...    'Close': [100, 105, 102, 108],
    ...    'Position': [0, 1, 1, 1],
    ...    'EntryPrice': [0, 105, 105, 105]
    ... })
    >>> df = apply_stop_loss_take_profit(df, stop_loss=0.03, take_profit=0.05)
    """
    if data.empty:
        logger.warning("Data is empty. No stop-loss/take-profit logic applied.")
        return data

    required_cols = [price_column, position_column, 'EntryPrice']
    for col in required_cols:
        if col not in data.columns:
            logger.error(f"Column '{col}' is missing in DataFrame.")
            raise KeyError(f"Column '{col}' is missing in DataFrame.")

    # We assume 'Position' is 1 if in a trade, 0 if not.  
    # (Adjust if your logic has short trades = -1 or partial trades, etc.)
    current_price = data[price_column]
    entry_price = data['EntryPrice']

    # Stop-loss check: if (current_price <= entry_price * (1 - stop_loss))
    sl_trigger = current_price <= entry_price * (1 - stop_loss)

    # Take-profit check: if (current_price >= entry_price * (1 + take_profit))
    tp_trigger = current_price >= entry_price * (1 + take_profit)

    # We only want to close positions where Position == 1
    # So the final trigger is positions that are open AND (stop-loss OR take-profit is hit).
    close_trigger = (data[position_column] == 1) & (sl_trigger | tp_trigger)

    # Close those positions by setting 'Position' back to 0
    data.loc[close_trigger, position_column] = 0

    # (Optional) If you want to log which rows got triggered:
    closed_indices = close_trigger[close_trigger == True].index
    if len(closed_indices) > 0:
        logger.info(f"Closed positions at indices: {closed_indices.tolist()}")

    return data

def example_trade_entry_logic(data: pd.DataFrame, price_column: str = 'Close') -> pd.DataFrame:
    """
    A simple placeholder for how you might set 'Position' to 1 (open a trade).
    This is purely illustrative—replace with your actual entry logic.

    For example, this might set Position=1 if the price is above a certain threshold,
    or if a signal says 'BUY'.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame that must have at least a 'Close' column.
    price_column : str, optional
        The name of the price column used in the logic.

    Returns
    -------
    pd.DataFrame
        DataFrame with updated 'Position' & 'EntryPrice' if a new position is opened.
    """
    if price_column not in data.columns:
        raise KeyError(f"Column '{price_column}' not found in DataFrame.")

    # For demonstration: buy if the price goes above 100
    data['Position'] = 0 if 'Position' not in data.columns else data['Position']
    data['EntryPrice'] = 0 if 'EntryPrice' not in data.columns else data['EntryPrice']

    new_trades = (data[price_column] > 100) & (data['Position'] == 0)
    data.loc[new_trades, 'Position'] = 1
    data.loc[new_trades, 'EntryPrice'] = data.loc[new_trades, price_column]

    return data


def example_trade_exit_logic(data: pd.DataFrame, price_column: str = 'Close') -> pd.DataFrame:
    """
    A simple placeholder for how you might set 'Position' to 0 (close a trade).
    This is purely illustrative—replace with your actual exit logic.

    For example, this might set Position=0 if the price is below a certain threshold,
    or if a signal says 'SELL'.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame that must have at least a 'Close' column.
    price_column : str, optional
        The name of the price column used in the logic.

    Returns
    -------
    pd.DataFrame
        DataFrame with updated 'Position'.
    """
    if price_column not in data.columns:
        raise KeyError(f"Column '{price_column}' not found in DataFrame.")

    # For demonstration: close if the price goes below 98
    exit_trades = (data[price_column] < 98) & (data['Position'] == 1)
    data.loc[exit_trades, 'Position'] = 0

    return data
