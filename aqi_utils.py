#-------------------------------------------------------------------------------
# Title: aqi_utils.py
# Copyright: Danaher Digital
# Time: 2021
# Desc: This script contains utilities for time series imputation.
#-------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats
from typing import List, Tuple

#-------------------------------------------------------------------------------
# measure_norm_stats
#-------------------------------------------------------------------------------
def measure_norm_stats(x: np.array, is_per_channel: bool=False) -> Tuple[float]:
    """
    Measures stats to be passed in as a parameter to the normalize() and
    unnormalize() methods.  They can optionally be computed on a per-channel
    basis, which is useful when channels greatly differ in character.

    :param x: The input data.
    :param is_per_channel: Whether to measure separate stats for each channel.
    :return: Returns the stats to pass in to the normalization methods.
    """
    if is_per_channel:
        num_axes = len(x.shape)
        axes = tuple(range(num_axes - 1))
        m = x.mean(axis=axes, keepdims=True)
        sd = x.std(axis=axes, keepdims=True)

        # Combine the Target's channels (1st, 2nd, and last):
        a = m[0][0]
        m_avg = (a[0] + a[1] + a[-1]) / 3.0
        m[0][0][0] = m[0][0][1] = m[0][0][-1] = m_avg
        
        a = sd[0][0]
        sd_avg = np.sqrt((a[0] ** 2 + a[1] ** 2 + a[-1] ** 2) / 3.0)
        sd[0][0][0] = sd[0][0][1] = sd[0][0][-1] = sd_avg

        sd[np.where(sd == 0)] = 1.0 # Handle all values identical.
        stats = (m, sd)
    else:
        sd = x.std()
        if sd == 0.0: sd = 1.0 # Handle all values identical.
        stats = (x.mean(), sd)
    return stats

#-------------------------------------------------------------------------------
# normalize
#-------------------------------------------------------------------------------
def normalize(x: np.array, norm_stats: tuple, is_per_channel: bool=True) \
    -> np.array:
    """
    Normalizes data for training an AI model.  The data is centererd by
    subtracting the mean, and then scaled by dividing by the standard deviation.
    
    :param x: The input data to normalize.
    :param norm_stats: The stats returned by measure_norm_stats().
    :param is_per_channel: Whether to measure separate stats for each channel.
    :return: Returns the normalized version of x.
    """
    if len(x.shape) < len(norm_stats[0].shape):
        m = norm_stats[0][0][0][0]
        sd = norm_stats[1][0][0][0]
        x = (x - m) / sd
    else:
        x = (x - norm_stats[0]) / norm_stats[1]
    return x

#-------------------------------------------------------------------------------
# unnormalize
#-------------------------------------------------------------------------------
def unnormalize(x: np.array, norm_stats: Tuple[float],
    is_per_channel: bool==True) -> np.array:
    """
    Un-normalizes data.
    
    :param x: The input data to normalize.
    :param norm_stats: The stats returned by measure_norm_stats().
    :param is_per_channel: Whether to measure separate stats for each channel.
    :return: Returns the normalized version of x.
    """
    if len(x.shape) == len(norm_stats[0].shape) - 1:
        m = norm_stats[0][0]
        sd = norm_stats[1][0]
        x = x * sd + m        
    elif len(x.shape) < len(norm_stats[0].shape):
        m = norm_stats[0][0][0][0]
        sd = norm_stats[1][0][0][0]
        x = x * sd + m        
    else:
        x = x * norm_stats[1] + norm_stats[0]
    return x

#-------------------------------------------------------------------------------
# smape
#-------------------------------------------------------------------------------
def smape(a: np.array, f: np.array, epsilon: float=1e-5) -> float:
    """
    Calculates the Symmetric Mean Absolute Percentage Error (SMAPE) metric
    between 2 arrays.
    https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error

    :param a: Actual values.
    :param f: Forecasted values.
    :return: Returns the SMAPE value.
    """
    len_a = len(a)
    if len_a == 0 or len_a != len(f):
        print('!!! ERROR array lengths must match !!!')
        return 0.0

    # Calculate the denominator first to check for division by 0:
    denom = np.abs(a) + np.abs(f)
    if np.all(denom > epsilon):
        # There are no 0 values.
        return 1.0 / len_a * np.sum(2.0 * np.abs(f - a) / denom * 100.0)
    else:
        # There are some 0 values, so remove them with a mask.
        mask = [denom[i] > epsilon for i in range(len(denom))]
        len_mask = np.count_nonzero(mask)
        if len_mask == 0:
            return 0.0 # All elements are near 0, so that is very similar.
        else:
            return 1.0 / len_mask * np.sum(2.0 * np.abs(f[mask] - a[mask]) \
                / denom[mask] * 100.0)

#-------------------------------------------------------------------------------
# compute_correlation
#-------------------------------------------------------------------------------
def compute_correlation(y_true: np.array, y_pred: np.array,
    epsilon: float=1e-5) -> float:
    """
    Computes the Pearson correlation coefficient between 2 arrays.\
    https://en.wikipedia.org/wiki/Pearson_correlation_coefficient

    :param y_true: Ground-truth values.
    :param y_pred: Predicted values.
    :return: Returns the coefficient, "r".
    """
    assert(len(y_true) > 0 and len(y_true) == len(y_pred)), \
        '!!! ERROR array lengths must match !!!'

    # Handle that we can't compute correlation on constant data:
    min_true = y_true.min()
    min_pred = y_pred.min()
    is_true_same = y_true.max() - min_true <= epsilon
    is_pred_same = y_pred.max() - min_pred <= epsilon
    if is_true_same or is_pred_same:
        if is_true_same and is_pred_same and abs(min_true - min_pred) <= epsilon:
            r = 1.0 # They are identical so that is perfect correlation.
        else:
            r = 0.0
    else:
        r, p = stats.pearsonr(y_true, y_pred)
    return r

#-------------------------------------------------------------------------------
# compute_metrics
#-------------------------------------------------------------------------------
def compute_metrics(y_true: np.array, y_pred: np.array) -> np.array:
    """
    Computes 4 different metrics for comparing 2 time series.

    :param true: Ground-truth values.
    :param pred: Predicted values.
    :return: Returns correlation, MAE, RMSE, Smape
    """
    r = compute_correlation(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    s = smape(y_true, y_pred)
    return np.array([r, mae, rmse, s])

#-------------------------------------------------------------------------------
# interpolate_nan
#-------------------------------------------------------------------------------
def interpolate_nan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolates the NaN values in a dataframe using cubic splines.

    :param df: The dataframe to interpolate:
    :return: Returns the interpolated dataframe.
    """
    # Interpolate (by cubic spline) any NAs in Corrected:
    values = df.Corrected.interpolate(method='spline', order=3).to_numpy()

    # Interpolation won't handle the ends, so fix those manually:
    n = len(values)
    b = np.isnan(values)

    # Search left end:
    for i in range(n):
        if not b[i]:
            v = values[i]
            break
    for j in range(i):
        values[j] = v

    # Search right end:
    for i in range(n):
        if not b[n-1 - i]:
            v = values[n-1 - i]
            break
    for j in range(i):
        values[n-1 - j] = v

    df.loc[:, 'Corrected'] = values
    return df

#-------------------------------------------------------------------------------
# resample_series
#-------------------------------------------------------------------------------
def resample_series(df_in: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """
    This method inserts any missing timepoints and imputes any gaps in the
    'Corrected' column of a given time series. It can be used to cleanup
    surrogate series so that they have all the same data points as the target.

    :param df_in: The input dataframe, which must have columns named 'Timestamp'.
    :return: Returns a modified dataframe.
    """
    # Check if the requisite columns exist:
    assert('timestamp' in df_in.columns and 'Corrected' in df_in.columns), \
        '!!! ERROR: Missing required columns !!!'

    # Insert any missing rows with NAs.  This also removes any drifted rows,
    # such as if the timestamp is a minute late.
    df_out = df_in.set_index('timestamp')
    df_out = df_out.asfreq(f'{minutes}Min').reset_index()

    # Replace any invalid values with NaNs:
    df_out.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Replace any NaNs with interpolated values:
    df_out = interpolate_nan(df_out)
    return df_out

#-------------------------------------------------------------------------------
# __main__
#-------------------------------------------------------------------------------
if __name__ == "__main__":

    # Test metrics:
    true = np.array([1, 2, 3, 4]).astype(float)
    pred = np.array([6, 7, 7, 6]).astype(float)
    r = compute_correlation(true, pred)
    print('True:', true, ', Pred:', pred, ', r:', r)