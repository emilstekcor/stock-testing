# modules/equity_curve.py
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def plot_equity_curve(equity_curve: pd.Series, title:str = "Equity Curve") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=equity_curve, mode='lines'))
    fig.update_layout(title=title, xaxis_title="Trades", yaxis_title="Cumulative Return")
    return fig
    
def plot_strategies_equity_curves(equity_curves: dict, title:str = "Strategies Equity Curves") -> go.Figure:
    fig = go.Figure()
    for strategy, curve in equity_curves.items():
        fig.add_trace(go.Scatter(y=curve, mode='lines', name=strategy))
    fig.update_layout(title=title, xaxis_title="Trades", yaxis_title="Cumulative Return")
    return fig

def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """Calculates the maximum drawdown of an equity curve.
    Args:
        equity_curve (pd.Series): The cumulative equity curve.
    Returns:
        float: The maximum drawdown as a percentage.
    """
    if equity_curve.empty:
        return 0.0
    peak = equity_curve.iloc[0]
    max_drawdown = 0
    for value in equity_curve:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    return max_drawdown

def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculates the Sortino ratio.
        Args:
           returns (pd.Series): daily returns of strategy
           risk_free_rate (float): risk free rate for this.
        Returns:
           float: sortino ratio (mean / std)
        """
    if returns.empty:
        return 0.0
    downside_returns = returns[returns < 0]
    if downside_returns.empty:
      return 0.0
    std_dev_downside = downside_returns.std()
    if std_dev_downside == 0:
        return 0.0
    return (returns.mean() - risk_free_rate) / std_dev_downside
