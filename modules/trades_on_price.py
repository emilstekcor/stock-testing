# modules/trades_on_price.py
import plotly.graph_objects as go
import pandas as pd

def plot_trades_on_price(data: pd.DataFrame, ticker:str, price_column = 'Close') -> go.Figure:
    """
     Plots trade markers over the stock close price.
    """
    fig = go.Figure()
    # Plot the close price
    fig.add_trace(go.Scatter(x=data.index, y=data[price_column], mode='lines', name=f'{price_column} Price'))

    # Find buy signals (where Position changes from 0 to 1)
    buy_signals = data[data['Position'].diff() == 1]
    # Plot buy points
    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals[price_column],
                             mode='markers', marker=dict(symbol='triangle-up', size=10, color='green'),
                             name='Buy'))
    # Find sell signals (where Position changes from 1 to 0)
    sell_signals = data[data['Position'].diff() == -1]
    # Plot sell points
    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals[price_column],
                             mode='markers', marker=dict(symbol='triangle-down', size=10, color='red'),
                             name='Sell'))
    fig.update_layout(title=f"Trades on {ticker}", xaxis_title="Date", yaxis_title="Price")
    return fig
