# stock-testing
just a python code I am working on to practice with


#Hyperparameter Optimization Trading Strategy

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/YOUR_USERNAME/YOUR_REPO_NAME)

This Streamlit application allows you to optimize and backtest trading strategies using historical stock data. It includes functionalities for single ticker analysis, bulk optimization of multiple tickers, various data transforms, noise injection, and more. It also integrates with OpenAI and news APIs for advanced features.

## Table of Contents
1.  [Introduction](#introduction)
2.  [Features](#features)
3.  [Getting Started](#getting-started)
    *   [Prerequisites](#prerequisites)
    *   [Installation](#installation)
    *   [Configuration](#configuration)
    *   [Running the Application](#running-the-application)
4.  [Usage](#usage)
    *   [Single Ticker Mode](#single-ticker-mode)
    *   [Multiple Tickers Mode](#multiple-tickers-mode)
    *   [Advanced Settings](#advanced-settings)
    *   [Saving Settings](#saving-settings)
5.  [Modules](#modules)
    *  [Data Fetching](#data-fetching)
    *  [Optimization](#optimization)
    *  [Results Handling](#results-handling)
    *  [UI Rendering](#ui-rendering)
    *  [Backtesting](#backtesting)
    *  [Trading Logic](#trading-logic)
    *  [Transforms](#transforms)
    *  [Noise Injection](#noise-injection)
    *  [Utilities](#utilities)
6.  [Technologies Used](#technologies-used)
7.  [Contributing](#contributing)
8.  [License](#license)
9.  [Disclaimer](#disclaimer)

## Introduction

This project provides a platform for exploring and optimizing various trading strategies. It utilizes libraries such as `Streamlit`, `pandas`, `optuna`, and `yfinance` to provide a robust environment for quantitative analysis. You can experiment with different trading parameters and evaluate the strategy's effectiveness through backtesting on simulated data. This also includes experimental support for incorporating sentiment analysis using APIs, and various technical analysis transforms.

## Features

*   **Single Ticker Optimization:** Optimize trading parameters for a single stock ticker.
*   **Multiple Tickers Optimization:** Run the optimization process on multiple tickers simultaneously.
*   **Data Fetching:** Uses `yfinance` (and custom providers) to fetch historical stock data.
*   **Hyperparameter Optimization:** Utilizes `optuna` for parameter tuning of trading strategies.
*   **Backtesting:** Evaluates strategy performance with customizable metrics on simulated real-time data.
*   **Data Transforms:** Includes technical analysis transformations such as causal filtering, FFT, STL decomposition and volatility indicators.
*   **Noise Injection:** Adds noise to the data to test strategy robustness.
*   **Sentiment Analysis (Experimental):** Integrates with OpenAI and news APIs for sentiment analysis.
*   **Visualization:** Displays results with interactive charts and tables in Streamlit.
*   **Configuration Management:** Saves and loads settings via a JSON config file.
*   **Detailed Logging:** Provides comprehensive logs for debugging and monitoring.
*   **Modern UI:** Uses Streamlit to provide a clean and interactive user interface.

## Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:

*   **Python 3.7+**
*   **pip** (Python package installer)

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
    cd YOUR_REPO_NAME
    ```

2.  **Install required Python packages:**

    ```bash
    pip install -r requirements.txt
    ```

    You may encounter issues with the `plotly` package that can often be solved by installing the `kaleido` package:

    ```bash
    pip install kaleido
    ```

3.  **Optional: API Keys**

    If you want to test the sentiment analysis features, sign up for an API key from the following services:
    *   [OpenAI](https://openai.com/api/)
    *   [News API](https://newsapi.org/)

    These API keys can be added within the "Advanced Settings" tab of the app.

### Configuration

1.  **(Optional) Create the `trading_config.json` File:**

    If a `trading_config.json` file does not already exist, the app will begin with default settings. You can create your own or modify the created config in the advanced settings of the app, and it will create / update the file itself. If you decide to make your own file, the JSON structure should look something like this, although you can omit any keys and the program will provide its own defaults. Note that this structure will be auto-generated after the first run, so you may simply decide to begin the app and modify from within it. The below config has all of the keys as a reference.

    ```json
    {
        "mode": "Single Ticker",
        "sampler_choice": "TPESampler",
        "n_trials": 100,
        "n_startup": 10,
        "cross_val_folds": 3,
        "detailed_logs": true,
        "soft_logic": false,
        "filter_enabled": false,
        "fft_enabled": false,
        "volatility_enabled": false,
        "stl_enabled": false,
        "use_noise": false,
        "noise_type": "white",
        "noise_intensity": 0.01,
        "ar_coeff": 0.5,
        "cyclic_frequency": 50,
        "distort_intensity": 0.1,
        "use_extreme_short": false,
        "plot_combined_curves": false,
        "openai_api_key": "",
        "news_api_key": "",
        "sentiment_weight": 0.25,
        "use_chatgpt": false,
        "use_news_api": false
    }
    ```

    The `trading_config.json` file saves settings and loads them on startup. If it does not exist, a default version of the settings will be used.

### Running the Application

1.  **Navigate to the project directory:**

    ```bash
    cd YOUR_REPO_NAME
    ```

2.  **Run the Streamlit app:**

    ```bash
    streamlit run main.py
    ```
    This will open the application in your default web browser.

## Usage

### Single Ticker Mode

1.  Select "Single Ticker" in the dropdown at the top.
2.  Enter the stock ticker in the `Ticker` input field (e.g. AAPL).
3.  Select start and end dates for the data.
4.  Select your preferred price column, such as `Close`, and entry price, such as `Open`.
5.  Optionally adjust the simulation steps to determine the length of the validation period.
6.  Choose the data source from the dropdown.
7.  Click on "Fetch & Show Data" to retrieve and show historical data.
8.  Click on "Start Optimization & Validation" to start the optimization process.
9.  View the results in the "Optimization" and "Validation" tabs.

### Multiple Tickers Mode

1.  Select "Multiple Tickers" in the dropdown at the top.
2.  Enter multiple tickers separated by commas in the text area (e.g., AAPL,MSFT,GOOGL).
3.  Select start and end dates for the data.
4.  Select your preferred price column, such as `Close`, and entry price, such as `Open`.
5.  Click "Run Bulk Optimization" to analyze all the listed tickers.
6.  View the overall results, including charts of all generated equity curves.

### Advanced Settings

Use the "Advanced Settings" collapsible panel to adjust the application parameters:

*   **Mode:** Choose between "Single Ticker" or "Multiple Tickers".
*   **Sampler:** Select the Optuna sampler (`TPESampler` or `RandomSampler`).
*   **Trials:** Set the number of optimization trials.
*   **Startup Trials:** Number of trials for the sampler warm-up.
*   **Cross-Validation Folds:** Set number of cross-validation folds.
*   **Detailed Logs:** Toggle detailed logging.
*   **Soft Logic:** Enable "soft logic" for parameters.
*   **Feature Selection:** Enable/disable data transformation features such as FFT, volatility, filtering and STL decomposition.
*   **Noise Options:** Enable noise injection and select noise type, intensity and other specific parameters for the noise.
*   **Extreme Short:** Toggle the option to allow extreme shorts in the backtesting logic.
*   **Combined Curves Plot:** Enable plotting of combined equity curves for the multiple tickers result.
*   **API Keys:** Enter your OpenAI and News API keys for sentiment analysis features, as well as select if you want to use these features.
*   **Sentiment Weight:** Adjust the weight for sentiment analysis in the backtest.

### Saving Settings

Click the "Save Current Settings" button to persist changes to the `trading_config.json` file. This enables you to load the same settings next time you run the app.

## Modules

This project is structured into several key modules. Below is a brief explanation of each:

### Data Fetching
*   `data_fetch.py`: Fetches market data using `yfinance`.

### Optimization
*   `optimizations.py`: Implements the Optuna hyperparameter optimization for the trading strategy. Contains both `run_optimization_for_ticker` which uses the "standard" strategy, and `run_optimization_for_ticker_soft` which is an experimental implementation that aims to smooth strategy performance for less choppy returns.

### Results Handling
*   `results.py`: Manages the display and rendering of optimization and validation results.

### UI Rendering
*   `ui.py`: Manages the rendering of sidebar options and related UI elements.

### Backtesting
*   `backtesting.py`: Contains backtesting functions using historical or simulated data.

### Trading Logic
*   `logic_normal.py`: Defines the standard trading strategy, based on a number of simple indicators.
*   `trades_logic.py`: Example trading entry and exit logic functions, for potential future expansion of the project.

### Transforms
*   `transforms.py`: Contains functions to apply data transforms, technical indicators and data de-noising. Transforms include: FFT, causal filtering, volatility indicators, and STL decomposition.

### Noise Injection
*   `noise.py`: Contains functions to inject noise into time series data. Noise injection includes: white noise, brownian noise, autoregressive noise, cyclic noise, as well as a function to generally distort a time series.

### Utilities
*   `main.py`: Contains the main logic of the Streamlit application, as well as other utilities including reading and writing config files, date calculations and simulating real-time data.

## Technologies Used

*   **Streamlit:** For building the interactive user interface.
*   **pandas:** For data manipulation and analysis.
*   **optuna:** For hyperparameter optimization.
*   **yfinance:** For fetching stock market data.
*   **NumPy:** For numerical computations.
*   **Plotly:** For interactive charting.
*   **JSON:** For configuration file handling.
*   **OpenAI:** For sentiment analysis using chatGPT (optional).
*   **News API:** For sentiment analysis based on news headlines (optional).
*   **Dateutil:** For date-related calculations.

## Contributing

Contributions are welcome! Please feel free to submit a pull request with any bug fixes, improvements, or new features.

1. Fork the repository.
2. Create a new branch for your feature (`git checkout -b feature/your-new-feature`).
3. Commit your changes (`git commit -am 'Add your new feature'`).
4. Push to the branch (`git push origin feature/your-new-feature`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This software is provided for educational purposes only. Trading and investing involve risks. Past performance is not indicative of future results. Please consult with a financial advisor before making investment decisions.
