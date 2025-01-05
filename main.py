# main.py

# ============================
# 1. Import Statements
# ============================
import streamlit as st
import pandas as pd
from datetime import datetime
import json
import os
import optuna
from typing import Dict, Any
import logging
import traceback
from dateutil.relativedelta import relativedelta
import warnings

# Import modules
from modules.data_fetch import fetch_data
from modules.optimizations import run_optimization_for_ticker, run_optimization_for_ticker_soft
from modules.results import display_optimization_results, render_optimization_history, display_validation_results, display_bulk_results
from modules.ui import render_sidebar_options, render_options_explanation
from modules.backtesting import backtest_with_metrics
from modules.logic_normal import objective
from modules.trades_logic import example_trade_entry_logic, example_trade_exit_logic
from modules.transforms import apply_causal_filter, apply_fft, calculate_volatility_indicator, apply_stl
from modules.noise import add_white_noise, add_brownian_noise, add_autoregressive_noise, add_cyclic_noise, distort_signal
# ============================
# 2. Logging Configuration
# ============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ============================
# 3. Utility Functions
# ============================

def load_settings(config_file: str) -> Dict[str, Any]:
    """
    Loads settings from a JSON configuration file.
    """
    if not os.path.exists(config_file):
        logger.warning(f"Config file {config_file} does not exist. Using default settings.")
        return {}
    with open(config_file, 'r') as f:
        settings = json.load(f)
    logger.info(f"Settings loaded from {config_file}.")
    return settings

def save_settings(settings: Dict[str, Any], config_file: str):
    """
    Saves settings to a JSON configuration file.
    """
    with open(config_file, 'w') as f:
        json.dump(settings, f, indent=4)
    logger.info(f"Settings saved to {config_file}.")

def calculate_start_date(current_date: datetime, years: int = 5) -> datetime:
    """
    Calculates the start date by subtracting a number of years from the current date.
    """
    return current_date - relativedelta(years=years)

def simulate_real_time_data(data: pd.DataFrame, steps: int = 200, initial_steps: int = 200) -> pd.DataFrame:
    """
    Simulates real-time data for backtesting purposes by performing a random walk on the 'close' price.
    """
    try:
        simulated_data = data.copy().tail(initial_steps).reset_index(drop=True)
        
        for _ in range(steps):
            last_close = simulated_data.iloc[-1]['close']
            # Simulate next close price with small random walk
            new_close = last_close * (1 + np.random.normal(0, 0.001))
            
            # For simplicity, set Open, High, Low to new_close
            new_row = {
                'open': new_close,
                'high': new_close,
                'low': new_close,
                'close': new_close
            }
            simulated_data = simulated_data.append(new_row, ignore_index=True)
        
        logger.info(f"Simulated real-time data: {simulated_data.shape}")
        return simulated_data
    except Exception as e:
        st.error(f"Error simulating real-time data: {e}")
        logger.error(f"Error simulating real-time data: {e}", exc_info=True)
        return pd.DataFrame()

# ============================
# 8. Main Function
# ============================

def main():
    # Title at the top
    st.markdown(
        """
        <h1 style='text-align: center;'>
            Kaiba’s Hyperparameter Optimization<br>
            <small style='font-size: 0.6em; color: #888;'>Trading Strategy</small>
        </h1>
        """,
        unsafe_allow_html=True
    )

    # Initialize session state variables if not present
    if 'data' not in st.session_state:
        st.session_state.data = None

    config_file = "trading_config.json"
    if 'settings' not in st.session_state:
        st.session_state.settings = {}
    if 'best_params' not in st.session_state:
        st.session_state.best_params = {}
    if 'best_value' not in st.session_state:
        st.session_state.best_value = None

    # Load settings
    try:
        st.session_state.settings = load_settings(config_file)
        st.write("Settings loaded successfully.")
    except Exception as e_load:
        st.error(f"Failed to load settings: {e_load}")
        st.write(traceback.format_exc())
        logger.error("Failed to load settings.", exc_info=True)
        st.session_state.settings = {}  # Fallback to empty settings

    # Make a small status area for user messages at top
    status_col = st.empty()

    # Provide a short introduction or instructions
    st.markdown("""
    <hr />
    <p style='font-size: 0.9em; color: #BBB; text-align: center;'>
        Adjust the settings in the collapsible "Advanced Settings" below, or just run with defaults.
    </p>
    <hr />
    """, unsafe_allow_html=True)

    # Place the side options into an expander for a more modern, minimal look
    with st.expander("Advanced Settings", expanded=False):
        status_text = status_col
        options = render_sidebar_options(st.session_state.settings, status_text)
    render_options_explanation()

    # Extract relevant options
    mode = options['mode']
    sampler_choice = options['sampler_choice']
    n_trials = options['n_trials']
    n_startup = options['n_startup']
    cross_val_folds = options['cross_val_folds']
    detailed_logs = options['detailed_logs']
    soft_logic = options['soft_logic']
    feature_selection = {
        'filter_enabled': options['filter_enabled'],
        'fft_enabled': options['fft_enabled'],
        'volatility_enabled': options['volatility_enabled'],
        'stl_enabled': options['stl_enabled'],
    }
    noise_params = {
        'use_noise': options['use_noise'],
        'noise_type': options['noise_type'],
        'noise_intensity': options['noise_intensity'],
        'ar_coeff': options['ar_coeff'],
        'cyclic_frequency': options['cyclic_frequency'],
        'distort_intensity': options['distort_intensity'],
    }
    extreme_short_params = {
        'use_extreme_short': options['use_extreme_short']
    }
    plot_combined_curves = options['plot_combined_curves']
    openai_api_key = options.get('openai_api_key', '')
    news_api_key = options.get('news_api_key', '')
    sentiment_weight = (
        options.get('sentiment_weight', 0.25)
        if isinstance(options.get('sentiment_weight', 0.25), float)
        else 0.25
    )
    use_chatgpt = options.get('use_chatgpt', False)
    use_news_api = options.get('use_news_api', False)

    # MAIN LOGIC
    if mode == "Single Ticker":
        st.subheader("Single Ticker Mode")
        col1, col2 = st.columns([1, 3], gap="medium")

        with col1:
            ticker = st.text_input("Ticker (Example: AAPL)", "AAPL").upper()
            start_date = st.date_input("Start Date", calculate_start_date(datetime.now()))
            end_date = st.date_input("End Date", datetime.now())
            price_column = st.selectbox(
                "Price Col",
                ['Open', 'Close', 'High', 'Low'],
                index=1  # Default to 'Close'
            )
            entry_price_type = st.selectbox(
                "Entry Price",
                ['Open', 'Close'],
                index=1  # Default to 'Close'
            )
            try:
                simulation_steps_local = st.number_input(
                    "Simulation steps",
                    min_value=100,
                    value=200,
                    max_value=2000  # Adjusted max_value for flexibility
                )
            except (ValueError, TypeError):
                simulation_steps_local = 200

            data_source = st.selectbox("Data Source", ['yfinance', 'my_second_level_provider'], index=0)

            if start_date >= end_date:
                st.error("Invalid date range: Start must be < End.")
                logger.error("Invalid date range: Start must be < End.")
                return

            # Button to fetch data
            if st.button("Fetch & Show Data"):
                st.write("Attempting data fetch...")
                st.write(f"Ticker: {ticker}, Start: {start_date}, End: {end_date}, Data Source: {data_source}")
                try:
                    st.session_state.data = fetch_data(
                        ticker,
                        start_date,
                        end_date,
                        data_source=data_source
                    )
                    if st.session_state.data.empty:
                        st.warning("No data loaded—data is empty.")
                        logger.warning("No data loaded. Possibly an invalid ticker or zero rows.")
                    else:
                        # Check for missing values
                        if st.session_state.data.isnull().values.any():
                            st.warning("Data contains missing values. These will be filled using forward fill.")
                            st.session_state.data = st.session_state.data.fillna(method='ffill').dropna()
                            st.write(f"Data after cleaning: {st.session_state.data.shape}")
                        
                        st.write(f"Data fetch completed. Shape: {st.session_state.data.shape}")
                        st.session_state.data_container = st.container()
                        with st.session_state.data_container:
                            st.write(st.session_state.data.tail())
                except Exception as e_fetch:
                    st.error(f"Data fetch failed with error: {e_fetch}")
                    st.write(traceback.format_exc())
                    logger.error("Data fetch failed.", exc_info=True)

            # Button to start optimization & validation
            if st.button("Start Optimization & Validation"):
                st.write("**Starting Optimization & Validation**")
                # Quick debug: check data shape
                if st.session_state.data is None or st.session_state.data.empty:
                    st.warning("No data loaded, please fetch data first.")
                    logger.warning("No data loaded, cannot proceed.")
                    return

                tabs = st.tabs(["Optimization", "Validation"])

                # --- Optimization Tab ---
                with tabs[0]:
                    st.write("**Optimization Tab Activated** (debug logs below)")
                    try:
                        st.write(f"Data shape for optimization: {st.session_state.data.shape}")
                        st.write(f"soft_logic = {soft_logic}")
                        if soft_logic:
                            st.write("Running run_optimization_for_ticker_soft...")
                            study = run_optimization_for_ticker_soft(
                                ticker=ticker,
                                data_f=st.session_state.data,
                                start_date=start_date,
                                end_date=end_date,
                                sampler_choice=sampler_choice,
                                n_trials=n_trials,
                                n_startup=n_startup,
                                cross_val_folds=cross_val_folds,
                                detailed_logs=detailed_logs
                            )
                        else:
                            st.write("Running run_optimization_for_ticker...")
                            study = run_optimization_for_ticker(
                                ticker=ticker,
                                data_f=st.session_state.data,
                                start_date=start_date,
                                end_date=end_date,
                                sampler_choice=sampler_choice,
                                n_trials=n_trials,
                                n_startup=n_startup,
                                cross_val_folds=cross_val_folds,
                                detailed_logs=detailed_logs,
                                feature_selection=feature_selection,
                                noise_params=noise_params,
                                extreme_short_params=extreme_short_params
                            )

                        if study is None:
                            st.error("Optimization returned None. Cannot proceed with validation.")
                            logger.error("No study object created. Possibly a crash in the optimization function.")
                        else:
                            st.write(f"**Optimization success**. Best Sharpe Ratio: {study.best_value}")
                            st.write(f"Best Params: {study.best_params}")
                            display_optimization_results(study, st.session_state.best_params if st.session_state.best_params else {})
                            render_optimization_history(study)

                    except Exception as e_opt:
                        st.error(f"Optimization crashed with error: {e_opt}")
                        st.write(traceback.format_exc())
                        logger.error("Optimization crashed.", exc_info=True)
                        study = None

                # --- Validation Tab ---
                with tabs[1]:
                    st.write("**Validation Tab Activated** (debug logs below)")
                    if not study:
                        st.write("No study found. Cannot validate.")
                    else:
                        try:
                            st.write("**Starting validation**...")
                            st.write(f"Using best params from study: {study.best_params}")
                            st.write(
                                f"Now calling simulate_real_time_data with steps={200}, "
                                f"initial_steps={simulation_steps_local}"
                            )
                            data_val = simulate_real_time_data(
                                st.session_state.data,
                                steps=200,
                                initial_steps=int(simulation_steps_local)
                            )
                            st.write(f"Validation dataset shape: {data_val.shape}")

                            st.write("Calling backtest_with_metrics...")
                            metrics = backtest_with_metrics(
                                data_val,
                                study.best_params,
                                detailed_logs,
                                soft_logic,
                                extreme_short_params['use_extreme_short'],
                                price_column=price_column,
                                entry_price_type=entry_price_type,
                                # Pass the new optional arguments
                                use_chatgpt=use_chatgpt,
                                use_news_api=use_news_api,
                                openai_api_key=openai_api_key,
                                news_api_key=news_api_key,
                                sentiment_weight=sentiment_weight
                            )
                            st.write("Backtest metrics returned. Keys:", list(metrics.keys()))
                            metrics['data'] = st.session_state.data

                            st.write("Displaying validation results now...")
                            display_validation_results(metrics, ticker)

                        except Exception as e_val:
                            st.error(f"Validation crashed with error: {e_val}")
                            st.write(traceback.format_exc())
                            logger.error("Validation crashed.", exc_info=True)

            # -- Column 2: Data Preview --
            with col2:
                st.subheader("Data Preview")
                if st.session_state.data is not None and not st.session_state.data.empty:
                    st.write(f"Data shape: {st.session_state.data.shape}")
                    st.write(st.session_state.data.head())
                else:
                    st.write("No data loaded yet.")

    else:
        # MULTIPLE TICKERS MODE
        st.subheader("Multiple Tickers Mode")
        st.write("Enter multiple tickers (comma-separated) below:")

        tickers_str = st.text_area("Tickers", "AAPL,MSFT,GOOGL")
        start_date = st.date_input("Start Date", calculate_start_date(datetime.now()))
        end_date = st.date_input("End Date", datetime.now())
        price_column = st.selectbox(
            "Price Col",
            ['Open', 'Close', 'High', 'Low'],
            index=1  # Default to 'Close'
        )
        entry_price_type = st.selectbox(
            "Entry Price",
            ['Open', 'Close'],
            index=1  # Default to 'Close'
        )

        if start_date >= end_date:
            st.error("Invalid date range: Start must be < End.")
            logger.error("Invalid date range: Start must be < End.")
            return

        if st.button("Run Bulk Optimization"):
            st.write("**Running Bulk Optimization**...")
            tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
            logger.info(f"Running Bulk Optimization for: {tickers}")
            st.write(f"Tickers to optimize: {tickers}")
            results = []
            equity_curves = {}

            for tkr in tickers:
                try:
                    st.write(f"Fetching data for: {tkr}")
                    data_f = fetch_data(tkr, start_date, end_date)
                    st.write(f"Data shape for {tkr}: {data_f.shape}")

                    if data_f.empty:
                        st.warning(f"Skipping {tkr} - no data loaded.")
                        logger.warning(f"Skipping {tkr} - data_f is empty.")
                        continue

                    if soft_logic:
                        st.write(f"{tkr}: run_optimization_for_ticker_soft...")
                        study = run_optimization_for_ticker_soft(
                            ticker=tkr,
                            data_f=data_f,
                            start_date=start_date,
                            end_date=end_date,
                            sampler_choice=sampler_choice,
                            n_trials=n_trials,
                            n_startup=n_startup,
                            cross_val_folds=cross_val_folds,
                            detailed_logs=detailed_logs
                        )
                        if study is None:
                            st.warning(f"{tkr} => No study returned. Skipping.")
                            logger.warning(f"{tkr} => No study returned. Skipping.")
                            continue
                        st.write(f"**{tkr} (Soft)** => Best Sharpe Ratio: {study.best_value:.5f}, params={study.best_params}")
                        best_params = study.best_params
                        data_val = simulate_real_time_data(data_f, steps=200, initial_steps=200)
                        st.write(f"Validation data for {tkr}: shape {data_val.shape}")
                        metrics = backtest_with_metrics(
                            data_val,
                            best_params,
                            detailed_logs,
                            soft_logic,
                            False,
                            price_column=price_column,
                            entry_price_type=entry_price_type,
                            # Pass the new optional arguments
                            use_chatgpt=use_chatgpt,
                            use_news_api=use_news_api,
                            openai_api_key=openai_api_key,
                            news_api_key=news_api_key,
                            sentiment_weight=sentiment_weight
                        )
                        st.write(f"Backtest metrics for {tkr} returned keys: {list(metrics.keys())}")
                        if "error" not in metrics:
                            equity_curves[f"{tkr} (Soft)"] = metrics['Equity Curve']
                        results.append((tkr, study.best_value, study.best_params))
                    else:
                        st.write(f"{tkr}: run_optimization_for_ticker...")
                        study = run_optimization_for_ticker(
                            ticker=tkr,
                            data_f=data_f,
                            start_date=start_date,
                            end_date=end_date,
                            sampler_choice=sampler_choice,
                            n_trials=n_trials,
                            n_startup=n_startup,
                            cross_val_folds=cross_val_folds,
                            detailed_logs=detailed_logs,
                            feature_selection=feature_selection,
                            noise_params=noise_params,
                            extreme_short_params=extreme_short_params
                        )
                        if study is None:
                            st.warning(f"{tkr} => No study returned. Skipping.")
                            logger.warning(f"{tkr} => No study returned. Skipping.")
                            continue
                        st.write(f"**{tkr}** => Best Sharpe Ratio: {study.best_value:.5f}, params={study.best_params}")
                        best_params = study.best_params
                        data_val = simulate_real_time_data(data_f, steps=200, initial_steps=200)
                        st.write(f"Validation data for {tkr}: shape {data_val.shape}")
                        metrics = backtest_with_metrics(
                            data_val,
                            best_params,
                            detailed_logs,
                            soft_logic,
                            extreme_short_params['use_extreme_short'],
                            price_column=price_column,
                            entry_price_type=entry_price_type,
                            # Pass the new optional arguments
                            use_chatgpt=use_chatgpt,
                            use_news_api=use_news_api,
                            openai_api_key=openai_api_key,
                            news_api_key=news_api_key,
                            sentiment_weight=sentiment_weight
                        )
                        st.write(f"Backtest metrics for {tkr} returned keys: {list(metrics.keys())}")
                        if "error" not in metrics:
                            equity_curves[f"{tkr}"] = metrics['Equity Curve']
                        results.append((tkr, study.best_value, study.best_params))

                except Exception as e_bulk:
                    st.error(f"Error optimizing {tkr}: {e_bulk}")
                    st.write(traceback.format_exc())
                    logger.error(f"Error optimizing {tkr}: {e_bulk}", exc_info=True)

            # Display all results
            display_bulk_results(results, equity_curves, plot_combined_curves)

    # Save Settings Button
    st.markdown("<hr/>", unsafe_allow_html=True)
    if st.button("Save Current Settings"):
        try:
            st.session_state.settings = options
            # No need to save indices anymore since all OHLC columns are being fetched
            st.session_state.settings["openai_api_key"] = options.get("openai_api_key", "")
            st.session_state.settings["news_api_key"] = options.get("news_api_key", "")
            st.session_state.settings["sentiment_weight"] = (
                options.get("sentiment_weight", 0.25)
                if isinstance(options.get('sentiment_weight', 0.25), float)
                else 0.25
            )
            st.session_state.settings["use_chatgpt"] = options.get("use_chatgpt", False)
            st.session_state.settings["use_news_api"] = options.get("use_news_api", False)

            save_settings(st.session_state.settings, config_file)
            st.success("Settings saved successfully.")
        except Exception as e_save:
            st.error(f"Failed to save settings: {e_save}")
            st.write(traceback.format_exc())
            logger.error("Failed to save settings.", exc_info=True)

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            main()
        except Exception as e_main:
            st.error(f"An unexpected error occurred: {e_main}")
            st.write(traceback.format_exc())
            logger.error("Unexpected error in main execution.", exc_info=True)
