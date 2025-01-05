import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any, List
import optuna.visualization as vis
import importlib.util
from modules.equity_curve import plot_equity_curve, plot_strategies_equity_curves
from modules.trades_on_price import plot_trades_on_price
import logging
import numpy as np
import scipy.signal as signal
import io
import base64

logger = logging.getLogger(__name__)

def _is_sklearn_installed() -> bool:
    return is_package_installed('sklearn')

def is_package_installed(package_name: str) -> bool:
    spec = importlib.util.find_spec(package_name)
    return spec is not None

def display_optimization_results(study, best_params):
    """Displays optimization results, using session state for best values/params."""
    if study is None:
        st.error("Optimization did not complete successfully.")
        return
    st.success("Optimization Done!")
    
    best_value = st.session_state.get('best_value', None)
    if best_value is not None:
      st.write(f"**Best Value (Sharpe)**: {best_value:.5f}")
    else:
       st.write("No Best Value Available")
    
    best_params = st.session_state.get('best_params', {})
    if best_params:
        st.write(f"**Best Params**: {best_params}")
    else:
       st.write("No Best Params Available")

def display_validation_results(metrics: Dict[str, Any], ticker: str = ""):
    """Displays validation/backtesting results."""
    if "error" in metrics:
        st.error(f"Validation backtest failed: {metrics['error']}")
        return

    st.subheader("Validation Results (Out-of-Sample)")
    st.write(f"**Validation Sharpe:** {metrics.get('sharpe', 0):.4f}")
    st.write(f"**Max Drawdown:** {metrics.get('max_drawdown', 0):.4f}")
    st.write(f"**Sortino Ratio:** {metrics.get('sortino', 0):.4f}")
    st.write(f"**Number of Trades:** {metrics.get('trades', 0):.0f}")

    st.subheader("Equity Curve")
    equity_curve = metrics.get('equity_curve', pd.Series())
    if not equity_curve.empty:
        fig = plot_equity_curve(equity_curve, title=f"Equity Curve - {ticker}")
        st.plotly_chart(fig, use_container_width=True)
    else:
         st.warning("Equity curve not available.")

    st.subheader("Trades On Price")
    trades_on_price_fig = metrics.get('trades_on_price', None)
    if trades_on_price_fig:
       st.plotly_chart(trades_on_price_fig, use_container_width=True)
    else:
         st.warning("Trades on price not available.")

    trades_details = metrics.get('trades_details', pd.DataFrame())
    if not trades_details.empty:
        st.subheader("Detailed Trade History")
        st.dataframe(trades_details)
    else:
         st.warning("Trade details not available.")

    data = metrics.get("data", pd.DataFrame())
    if not data.empty:
      st.subheader("Interactive Candlestick Chart")
      fig_candle = plot_interactive_candlestick(data, ticker, trades_details)
      st.plotly_chart(fig_candle, use_container_width=True)

      st.subheader("3D Price Landscape")
      fig_3d = plot_3d_price_landscape(data, ticker)
      st.plotly_chart(fig_3d, use_container_width=True)
      
      st.subheader("Sound of the Market")
      audio_bytes = generate_market_sound(data['Close'])
      st.audio(audio_bytes, format="audio/wav")
      
      st.subheader("Market Comparison")
      fig_comparison = plot_market_comparison(data, equity_curve, ticker)
      st.plotly_chart(fig_comparison, use_container_width=True)
    else:
      st.warning("Additional Data or plots are not available.")
def display_bulk_results(results: List[Any], equity_curves: Dict[str, pd.Series], plot_combined_curves: bool = False):
    """Displays bulk optimization results."""
    st.write("### Final Bulk Results")
    for tkr, val, prm in results:
        st.write(f"**{tkr}**: Sharpe={val:.5f}, Params={prm}")

    if equity_curves:
        st.subheader("Individual Equity Curves")
        for tkr, curve in equity_curves.items():
            fig = plot_equity_curve(curve, title=f"Equity Curve - {tkr}")
            st.plotly_chart(fig, use_container_width=True)
    else:
      st.warning("No individual equity curves found.")

    if plot_combined_curves and equity_curves:
        st.subheader("Combined Equity Curves")
        fig = plot_strategies_equity_curves(equity_curves, title="Combined Equity Curves")
        st.plotly_chart(fig, use_container_width=True)
    elif plot_combined_curves:
      st.warning("No combined equity curves found since it's off. Check the checkmark.")

def render_optimization_history(study):
    """Renders the optimization history and parameter importance plots."""
    if _is_sklearn_installed() and study is not None:
        st.subheader("Optimization History")
        try:
            fig_hist = vis.plot_optimization_history(study)
            st.plotly_chart(fig_hist, use_container_width=True)
        except ImportError as e:
            st.warning(f"Could not plot history; scikit-learn missing. {e}")
        
        st.subheader("Param Importances")
        try:
            fig_imp = vis.plot_param_importances(study)
            st.plotly_chart(fig_imp, use_container_width=True)
        except ImportError as e:
             st.warning(f"Could not plot param importances; scikit-learn missing. {e}")
    elif not _is_sklearn_installed():
        st.info("Install scikit-learn to see advanced Optuna plots.")
    elif study is None:
        st.warning("No Optimization Study to display history for.")
def plot_interactive_candlestick(data: pd.DataFrame, ticker: str, trades: pd.DataFrame = pd.DataFrame()) -> go.Figure:
    """Plots an interactive candlestick chart."""
    fig = go.Figure(data=[go.Candlestick(x=data.index,
                                        open=data['Open'],
                                        high=data['High'],
                                        low=data['Low'],
                                        close=data['Close'])])
    # Add buy/sell markers from the trade history
    if not trades.empty:
        buy_signals = trades[trades['Signal_RM'] == 1]
        sell_signals = trades[trades['Signal_RM'] == -1]
        fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Entry_Price'],
                                 mode='markers', marker=dict(symbol='triangle-up', size=10, color='green'),
                                 name='Buy'))
        fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Entry_Price'],
                                 mode='markers', marker=dict(symbol='triangle-down', size=10, color='red'),
                                 name='Sell'))
    fig.update_layout(title=f"Candlestick Chart for {ticker}",
                      xaxis_title="Date",
                      yaxis_title="Price",
                      xaxis_rangeslider_visible=False)
    return fig

def plot_3d_price_landscape(data: pd.DataFrame, ticker: str) -> go.Figure:
    """Plots a 3D surface plot of price data."""
    if data.empty:
      return go.Figure()
    time_points = np.arange(len(data))
    price_values = data['Close'].values
    volatility_values = data['Close'].rolling(window=10, min_periods=1).std().values
    fig = go.Figure(data=[go.Surface(z=np.array([price_values, volatility_values]), x=time_points, y=[0, 1])])
    fig.update_layout(title=f"3D Price Landscape for {ticker}",
                      scene=dict(xaxis_title="Time",
                                 yaxis_title="Volatility / Price",
                                 zaxis_title="Value"))
    return fig

def generate_market_sound(price_data: pd.Series, sample_rate: int = 44100, duration: float = 2.0) -> bytes:
    """Generates a sound based on price changes."""
    if price_data.empty:
      return b''
    price_changes = price_data.diff().dropna()
    frequencies = 440 + 200 * np.tanh(price_changes)
    t = np.linspace(0, duration, len(frequencies) * sample_rate, endpoint=False)
    tone = np.sin(2 * np.pi * np.repeat(frequencies, sample_rate) * t)
    tone = (tone * 32767).astype(np.int16)
    wav_file = io.BytesIO()
    signal.wavfile.write(wav_file, sample_rate, tone)
    wav_file.seek(0)
    return wav_file.read()

def plot_market_comparison(data: pd.DataFrame, equity_curve: pd.Series, ticker: str) -> go.Figure:
    """Plots the equity curve against the market return."""
    if data.empty or equity_curve.empty:
      return go.Figure()
    market_return = data['Close'].pct_change().fillna(0)
    market_cumulative_return = (1 + market_return).cumprod()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve, mode='lines', name='Strategy Equity Curve'))
    fig.add_trace(go.Scatter(x=market_cumulative_return.index, y=market_cumulative_return, mode='lines', name='Market Cumulative Return'))
    fig.update_layout(title=f"Strategy vs Market Comparison for {ticker}",
                      xaxis_title="Trades",
                      yaxis_title="Cumulative Return")
    return fig
