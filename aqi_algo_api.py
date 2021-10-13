#------------------------------------------------------------------------------
# Title: aqi_algo_api.py
# Copyright: Danaher Digital
# Time: 2021
# Desc: This script creates the API methods for a service to call.
#------------------------------------------------------------------------------
import pandas as pd
from aqi_data import derive_nn_samples, derive_period
from aqi_impute_gap import impute_gap
from typing import List, Tuple
from datetime import datetime

#-------------------------------------------------------------------------------
# algo_request
#-------------------------------------------------------------------------------
def algo_request(parameter: str, gap: List[datetime], interval: int,
    max_days: List[int]=[65, 2], minmax_nn_samples: List[int]=[10, 60],
    reg_samples: int=3, verbose=1) -> List[datetime]:
    """
    This method is the 1st algorithm call, and it requests which gap to impute.
    
    :param parameter: The type of sensor, such as GH, WT, DO, PH, SC, etc.
    :param period: List of the 2 timepoints of the gap, inclusive.
    :param interval: Time interval between sensor readings, in minutes.
    :param max_days: List of the maximum days to look backward/forward from gap.
    :param minmax_nn_samples: Min and max samples for training neural network.
    :param reg_samples: The number of samples for training Bayesian regression.
    :return: Returns list of timepoints demarking the period of data needed.
    """
    # Derive the number of training samples we can use for this gap:
    nn_samples = derive_nn_samples(interval, gap, max_days,
        minmax_nn_samples, reg_samples, verbose=1)

    # Derive the period of interest, big enough for regression and imputation:
    period, period_samples = derive_period(interval, gap, nn_samples,
        reg_samples, verbose=1)
    if verbose: print(f'Period containing {period_samples} samples: {period[0]} - {period[1]}.')
    return period   

#-------------------------------------------------------------------------------
# algo_impute
#-------------------------------------------------------------------------------
def algo_impute(parameter: str, gap: List[datetime], interval: int,
    df_data: pd.DataFrame, max_days: List[int]=[65, 2],
    minmax_nn_samples: List[int]=[10, 60], reg_samples: int=3, verbose: int=1) \
    -> Tuple[pd.DataFrame, str]:
    """
    This is the 2nd algorithm call, which performs imputation.

    :param parameter: The type of sensor, such as GH, WT, DO, PH, SC, etc.
    :param gap: List of the 2 timepoints of the gap, inclusive.
    :param interval: Time interval between sensor readings, in minutes.
    :param df_data: Dataframe whose 1st column is datetime, and rest are
        the target and surrogates.
    :param max_days: List of the maximum days to look backward/forward from gap.
    :param minmax_nn_samples: Min and max samples for training neural network.
    :param reg_samples: The number of samples for training Bayesian regression.
    :return: Returns a dataframe with columns for timestamps, prediction, and
        confidence band.  Also returns the name of the algorithm used, which is
        one of {'CNN', 'Linear'}.
    """
    # Impute the gap:
    y_true, y_pred, y_reg, band, metrics_reg, metrics_pred, algo, nn_samples = \
        impute_gap(df_data, interval, parameter, gap, max_days,
        minmax_nn_samples, reg_samples, verbose=1)

    # Construct a dataframe of results inside the gap:
    df_imputed = df_data[df_data.timestamp.between(gap[0], gap[1])].copy()
    cols = list(df_imputed.columns)
    cols.remove('timestamp')
    df_imputed.drop(cols, axis=1, inplace=True)
    df_imputed.loc[:, 'Predicted'] = y_pred
    df_imputed.loc[:, 'Confidence'] = band
    if verbose:
        print('Imputed Target DF shape (rows, cols) =', df_imputed.shape)
        print(df_imputed.head())
    return df_imputed, algo, metrics_reg, metrics_pred
