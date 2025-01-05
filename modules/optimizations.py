# modules/optimizations.py
import optuna
from typing import Dict, Any
import logging
import traceback
import pandas as pd
from datetime import datetime
import streamlit as st
import threading
from modules.logic_normal import objective, run_trial_fn_factory
from modules.logic_soft import objective_with_soft_logic, run_trial_fn_factory_soft
import os
import sqlite3
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
logger = logging.getLogger("OptunaApp")
SAMPLER_MAP = {
    'TPE': optuna.samplers.TPESampler,
    'CMA-ES': optuna.samplers.CmaEsSampler,
    'Random': optuna.samplers.RandomSampler
}
DATABASE_URL = "sqlite:///my_async_optuna.db"
pool = QueuePool(lambda: create_engine(DATABASE_URL), max_overflow=20)
def async_db_task(sql, params):
  with pool.connect() as conn:
        return conn.execute(text(sql), params).fetchall()
def run_optimization_for_ticker(
    ticker: str,
    data_f: pd.DataFrame,
    start_date: datetime,
    end_date: datetime,
    sampler_choice: str,
    n_trials: int,
    n_startup: int,
    cross_val_folds: int,
    detailed_logs: bool,
    feature_selection: Dict[str, bool],
    noise_params: Dict[str, Any],
    extreme_short_params: Dict[str,bool]
):
    best_params = {}
    best_val = None
    storage_url = f"sqlite:///{ticker}_refined.db"
    sampler_cls = SAMPLER_MAP[sampler_choice]
    if sampler_choice == "TPE":
        sampler = sampler_cls(n_startup_trials=int(n_startup), seed=42)
    else:
        sampler = sampler_cls(seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10)
    study_name = f"{ticker}_refined_study"
    study = optuna.create_study(
         study_name=study_name,
         storage=storage_url,
        load_if_exists=True,
        direction='maximize',
        sampler=sampler,
       pruner=pruner
     )
    run_trial_fn = run_trial_fn_factory(
        data_f,
       cross_val_folds,
        detailed_logs
    )
    def run_trial_fn_set_user_attrs(trial:optuna.trial.Trial) -> float:
        trial.set_user_attr('feature_selection', feature_selection)
        trial.set_user_attr('noise_params', noise_params)
        trial.set_user_attr('extreme_short_params', extreme_short_params)
        return run_trial_fn(trial)
    with st.spinner(f"Optimizing {ticker}..."):
        try:
            study.optimize(run_trial_fn_set_user_attrs, n_trials=int(n_trials), n_jobs=-1, show_progress_bar=True)
            if study is not None:
                best_val = study.best_value
                best_params = study.best_params
        except Exception as e:
           logger.error(f"Exception in optimization: {e} \n {traceback.format_exc()}")
           st.error(f"Optimization failed for {ticker}. Check logs for more details.")
           return None
    if best_val is not None:
       logger.info(f"Ticker={ticker}, Best Sharpe={best_val:.5f}, Params={best_params}")
    else:
       logger.info(f"Ticker={ticker} Optimization Failed. ")
    return study
def run_optimization_for_ticker_soft(
    ticker: str,
    data_f: pd.DataFrame,
    start_date: datetime,
    end_date: datetime,
    sampler_choice: str,
    n_trials: int,
    n_startup: int,
    cross_val_folds: int,
    detailed_logs: bool
):
    storage_url = f"sqlite:///{ticker}_soft.db"
    sampler_cls = SAMPLER_MAP[sampler_choice]
    if sampler_choice == "TPE":
        sampler = sampler_cls(n_startup_trials=int(n_startup), seed=42)
    else:
        sampler = sampler_cls(seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10)
    study_name = f"{ticker}_soft_study"
    study = optuna.create_study(
        study_name=study_name,
       storage=storage_url,
        load_if_exists=True,
       direction='maximize',
        sampler=sampler,
        pruner=pruner
     )
    run_trial_fn = run_trial_fn_factory_soft(
        data_f,
        cross_val_folds,
       detailed_logs
   )
    best_val = None
    best_params = {}
    with st.spinner(f"Optimizing {ticker} with Soft Logic..."):
         try:
           study.optimize(run_trial_fn, n_trials=int(n_trials), n_jobs=-1, show_progress_bar=True)
           if study is not None:
                 best_val = study.best_value
                 best_params = study.best_params
         except Exception as e:
           logger.error(f"Exception in optimization (soft logic): {e} \n {traceback.format_exc()}")
           st.error(f"Optimization failed for {ticker} (soft). Check logs for more details.")
           return None
    if best_val is not None:
      logger.info(f"Ticker={ticker}, Best Sharpe={best_val:.5f}, Params={best_params}")
    else:
         logger.info(f"Ticker={ticker} Optimization Failed. ")
    return study
