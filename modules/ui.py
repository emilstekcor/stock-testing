import streamlit as st
from datetime import datetime
import logging
from typing import Dict, Any
import optuna

logger = logging.getLogger(__name__)

DEFAULT_N_TRIALS = 500
DEFAULT_N_STARTUP = 50
DEFAULT_CROSS_VAL_FOLDS = 1

SAMPLER_MAP = {
    'TPE': optuna.samplers.TPESampler,
    'CMA-ES': optuna.samplers.CmaEsSampler,
    'Random': optuna.samplers.RandomSampler
}

def render_sidebar_options(settings: Dict[str, Any] = None, status_text=st.empty()) -> Dict[str, Any]:
    """
    Renders the Streamlit sidebar with user-adjustable settings.
    Returns a dict of updated settings from user input.
    """
    if settings is None:
        settings = {}

    with st.sidebar:
        st.title("Settings")
        
        # Mode
        mode_index = 0
        if isinstance(settings.get('mode'), int):
            mode_index = settings.get('mode', 0)
        mode = st.radio("Mode", ["Single Ticker", "Multiple Tickers"], index=mode_index)
        status_text.text(f"Mode: {mode}")

        # Optimization Settings
        with st.expander("Optimization Settings", expanded=True):
            sampler_keys = list(SAMPLER_MAP.keys())
            sampler_choice_idx = settings.get('sampler_choice', 0)
            if not isinstance(sampler_choice_idx, int):
                sampler_choice_idx = 0
            sampler_choice = st.selectbox("Sampler", sampler_keys, index=sampler_choice_idx)
            status_text.text(f"Sampler changed to: {sampler_choice}")

            n_trials = st.number_input(
                "Number of Trials",
                min_value=50, max_value=3000,
                value=int(settings.get('n_trials', DEFAULT_N_TRIALS)),
                step=50
            )
            n_startup = st.number_input(
                "TPE Startup Trials", 5, 300,
                value=int(settings.get('n_startup', DEFAULT_N_STARTUP)),
                step=5
            )
            cross_val_folds = st.number_input(
                "Cross-Val Folds (1=none)", 1, 5,
                value=int(settings.get('cross_val_folds', DEFAULT_CROSS_VAL_FOLDS))
            )
            detailed_logs = st.checkbox(
                "Detailed Logs",
                value=bool(settings.get('detailed_logs', False))
            )
            soft_logic = st.checkbox(
                "Use LLR / Soft Logic",
                value=bool(settings.get('soft_logic', False))
            )

        # Feature Selection
        with st.expander("Feature Selection", expanded=False):
            filter_enabled = st.checkbox("Causal Filter", value=bool(settings.get('filter_enabled', False)))
            fft_enabled = st.checkbox("FFT", value=bool(settings.get('fft_enabled', False)))
            volatility_enabled = st.checkbox("Volatility Indicator", value=bool(settings.get('volatility_enabled', False)))
            stl_enabled = st.checkbox("STL Decomposition", value=bool(settings.get('stl_enabled', False)))

        # Noise Addition
        with st.expander("Noise Addition", expanded=False):
            use_noise = st.checkbox("Add Noise", value=bool(settings.get('use_noise', False)))
            try:
                noise_type_idx = int(settings.get('noise_type_idx', 0))
            except (ValueError, TypeError):
                noise_type_idx = 0
            noise_type = st.selectbox(
                "Noise Type",
                ["white", "brownian", "autoregressive", "cyclic", "distort_ma"],
                index=noise_type_idx,
                disabled=not use_noise
            )
            noise_intensity = st.slider(
                "Noise Intensity", 0.0, 0.5,
                value=float(settings.get('noise_intensity', 0.05)),
                step=0.01,
                disabled=not use_noise
            )
            ar_coeff = st.slider(
                "AR Coefficient", 0.0, 1.0,
                value=float(settings.get('ar_coeff', 0.5)),
                step=0.1,
                disabled=(not use_noise or noise_type != "autoregressive")
            )
            cyclic_frequency = st.slider(
                "Cyclic Frequency", 0.01, 0.2,
                value=float(settings.get('cyclic_frequency', 0.1)),
                step=0.01,
                disabled=(not use_noise or noise_type != "cyclic")
            )
            distort_intensity = st.slider(
                "Distortion Intensity", 0.0, 0.2,
                value=float(settings.get('distort_intensity', 0.05)),
                step=0.01,
                disabled=(not use_noise or noise_type != "distort_ma")
            )

        # Extreme Short
        with st.expander("Extreme Short Trading", expanded=False):
            use_extreme_short = st.checkbox(
                "Use Extreme Short Strategy",
                value=bool(settings.get('use_extreme_short', False))
            )

        # Plotting
        with st.expander("Plotting", expanded=False):
            plot_combined_curves = st.checkbox(
                "Plot combined equity curves",
                value=bool(settings.get('plot_combined_curves', False))
            )

        # Additional Keys
        openai_api_key = st.text_input("OpenAI API Key", value=settings.get('openai_api_key', ''))
        news_api_key = st.text_input("News API Key", value=settings.get('news_api_key', ''))
        sentiment_weight = st.slider(
            "Sentiment Weight", 
            0.0, 1.0, 
            value=float(settings.get('sentiment_weight', 0.25)), 
            step=0.05
        )

        # Reset Trials
        reset_count = int(settings.get("reset_count", 0))
        if st.button("Reset Trials Count", key=f"reset_button_{reset_count}", help="Resets trials count."):
            settings['trial_count'] = 0
            settings['reset_count'] = reset_count + 1
            status_text.text('Number of Trials were reset!')
            st.rerun()

    return {
        "mode": mode,
        "sampler_choice": sampler_choice,
        "n_trials": n_trials,
        "n_startup": n_startup,
        "cross_val_folds": cross_val_folds,
        "detailed_logs": detailed_logs,
        "soft_logic": soft_logic,
        "filter_enabled": filter_enabled,
        "fft_enabled": fft_enabled,
        "volatility_enabled": volatility_enabled,
        "stl_enabled": stl_enabled,
        "use_noise": use_noise,
        "noise_type": noise_type,
        "noise_intensity": noise_intensity,
        "ar_coeff": ar_coeff,
        "cyclic_frequency": cyclic_frequency,
        "distort_intensity": distort_intensity,
        "use_extreme_short": use_extreme_short,
        "plot_combined_curves": plot_combined_curves,
        "noise_type_idx": noise_type_idx if use_noise else 0,
        "openai_api_key": openai_api_key,
        "news_api_key": news_api_key,
        "sentiment_weight": sentiment_weight,
        "reset_count": reset_count
    }

def render_options_explanation():
    st.write("""
    ## Options Explanation
    - **Mode**: Single Ticker or Multiple Tickers
    - **Sampler**: TPE, CMA-ES, or Random
    - **Number of Trials**: total trial count
    - **TPE Startup Trials**: how many random draws before TPE distribution
    - **Cross-Val Folds**: naive time-split folds
    - **Detailed Logs**: logs partial info
    - **Use LLR / Soft Logic**: If checked, uses the new 'soft logic' approach for signals
    - **Feature Selection**: Toggles filtering, FFT, Volatility, STL
    - **Noise Addition**: Adds random noise, AR, cyclic, or distortion 
    - **Extreme Short Trading**: Experimental short strategy
    - **Plot combined equity curves**: For visual comparison across tickers
    - **OpenAI/News API Keys** and **Sentiment Weight**: For optional AI-driven sentiment
    - **Reset Trials**: Resets the internal trial count to zero
    """)
