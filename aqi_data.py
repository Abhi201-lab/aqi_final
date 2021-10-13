#-------------------------------------------------------------------------------
# Title: aqi_data.py
# Copyright: Danaher Digital
# Time: 2021
# Desc: This scripts manipulates dataframes to make numpy arrays for training.
# and prediction.
#-------------------------------------------------------------------------------
from os.path import join
import numpy as np
import pandas as pd
import time
import yaml
from collections import Counter
from datetime import datetime, timedelta
import random
from typing import List, Tuple
import logging
import warnings
from aqi_utils import measure_norm_stats, normalize, resample_series
import tensorflow as tf
warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(level=logging.INFO) # Change INFO to DEBUG to diagnose.

#-------------------------------------------------------------------------------
# derive_nn_samples
#-------------------------------------------------------------------------------
def derive_nn_samples(interval_mins: int, gap: List[datetime],
    max_days: List[int]=[68, 4], minmax_nn_samples: List[int]=[10, 60],
    reg_samples: int=3, verbose=1) -> int:
    """
    This method derives the number of neural network training/validation samples
    to use for this gap.  It will try to use the maximum allowed number unless
    that would violate the constraint to use no more than a maximum number of
    days of data.

    :param interval_mins: Time between sensor readings, in minutes.
    :param gap: List of timestamps demarking the gap's endpoints.
    :param max_days: List of the maximum days to look backward/forward from gap.
    :param minmax_nn_samples: Min and max samples for training neural network.
    :param reg_samples: Number of samples to use for training linear regression.
    :return: Returns the number of training samples to use
    """
    pre_limit, post_limit = max_days
    min_nn, max_nn = minmax_nn_samples
    pre_samples = 1
    post_samples = 1

    # Measure duration of gap:
    gap_mins = int((gap[1] - gap[0]).total_seconds() // 60)
    gap_pnts = gap_mins // interval_mins + 1
    gap_mins = gap_pnts * interval_mins
    gap_hours = gap_mins / 60
    if verbose: print(f'Gap is {gap_mins} minutes, {gap_hours:.1f} hours, or {gap_pnts} points.')

    # Determine ideal number of samples:
    nn_samples = max_nn
    samples_before_gap = reg_samples + pre_samples + nn_samples + post_samples

    # Check if that's too many:
    pre_mins = samples_before_gap * gap_mins
    max_mins = pre_limit * (24 * 60)
    if pre_mins > max_mins:
        # Work backwards to derive the number of samples:
        samples_before_gap = int(max_mins // gap_mins)
        nn_samples = samples_before_gap - (reg_samples + pre_samples + post_samples)
        
        # If it's not enough, then punt to linear regression:
        if nn_samples < min_nn:
            nn_samples = 0
        
    return nn_samples

#-------------------------------------------------------------------------------
# derive_period
#-------------------------------------------------------------------------------
def derive_period(interval_mins: int, gap: List[datetime], nn_samples: int=60,
    reg_samples: int=3, verbose: int=1) -> Tuple[List[datetime], int]:
    """
    This method calculates the period of time needed to impute the requested gap.

    :param interval_mins: Time between sensor readings, in minutes.
    :param gap: List of timestamps demarking the gap's endpoints.
    :param nn_samples: Number of samples to use for training/validating network.
        NOTE: Set this number to be <= 0 for linear regression only.
    :param reg_samples: Number of samples to use for training linear regression.
    :return: Returns a list of endpoints demarking the period, and also the
        number of samples the period contains.
    """
    pre_samples = 1
    post_samples = 1
    boundary_pnts = 3

    # Measure duration of gap:
    gap_mins = int((gap[1] - gap[0]).total_seconds() // 60)
    gap_pnts = gap_mins // interval_mins + 1
    gap_mins = gap_pnts * interval_mins
    gap_hours = gap_mins // 60
    if verbose: print(f'Gap is {gap_mins} minutes, {gap_hours:.1f} hours, or {gap_pnts} points.')

    # Determine number of samples:
    if nn_samples <= 0:
        # Linear regression only.
        samples_before_gap = reg_samples
        samples_after_gap = 0
    else:
        samples_before_gap = reg_samples + pre_samples + nn_samples + post_samples
        samples_after_gap = post_samples
    num_samples = samples_before_gap + 1 + samples_after_gap
    if verbose: print('Samples before and after gap:', samples_before_gap, \
        ',', samples_after_gap, ', Total:', num_samples)

    # Derive period length:
    pre_mins = samples_before_gap * gap_mins
    post_mins = samples_after_gap * gap_mins + boundary_pnts * interval_mins
    period = [gap[0] - timedelta(minutes=pre_mins),
        gap[1] + timedelta(minutes=post_mins)]
    if verbose: print('Period:', period[0], '-', period[1])

    return period, num_samples

#-------------------------------------------------------------------------------
# check_window_boundaries
#-------------------------------------------------------------------------------
def check_window_boundaries(df_wnd: pd.DataFrame, window: List[datetime],
    desc: str, interval: int, verbose: int=1):
        
    t1 = df_wnd.timestamp.iloc[0]
    t2 = df_wnd.timestamp.iloc[len(df_wnd)-1]
    if verbose: print('T1:', window[0], t1, ', T2:', window[1], t2)
    
    sec_slack = interval * 60
    s1 = t1 > window[0] if (t1 - window[0]).seconds else (window[0] - t1).seconds
    s2 = t2 > window[1] if (t2 - window[1]).seconds else (window[1] - t2).seconds

    assert (s1 < sec_slack and s2 < sec_slack), \
        f'{desc} missing boundary points!!! DF: {t1} - {t2}, Window: {window[0]} - {window[1]}'

#-------------------------------------------------------------------------------
# extract_window
#-------------------------------------------------------------------------------
def extract_window(df: pd.DataFrame, window: List[datetime], desc: str=None,
    interval: int=None, is_check: bool=False, verbose: int=1) -> pd.DataFrame:
    """
    Extracts the rows of a dataframe coinciding with a certain window of time.
    
    :param df_in: The input dataframe from which to extract.
    :param window: The list of timestamps demarking the window to extract.
    :param desc: An optional description to print revealing the type of window.
    :param interval: Time between sensor readings, in minutes.
    :return: Returns a dataframe which is a subset of the input.
    """
    # Extract window:
    df_wnd = df[df.timestamp.between(window[0], window[1])].copy()
    if verbose: print('Window DF shape (rows, cols) =', df_wnd.shape)

    # Optionally check whether boundary points exist:
    if is_check and interval is not None:
        check_window_boundaries(df_wnd, window, desc, interval, verbose)

    return df_wnd

#-------------------------------------------------------------------------------
# extract_window_as_arrays
#-------------------------------------------------------------------------------
def extract_window_as_arrays(df_data: pd.DataFrame, window: List[datetime],
    is_reg: bool, interval: int=None, is_check: bool=False, verbose: int=1) \
    -> tuple:
    """
    Extracts the rows of a dataframe coinciding with a certain window of time,
    and returns them in the form of numpy arrays for target and surrogates.

    :param df_data: The dataframe of target and surrogates from which to extract.
    :param window: The list of timestamps demarking the window of time.
    :param is_reg: Whether to also extract regression's output.
    :param interval: Time between sensor readings, in minutes.
    :return: Returns a numpy array for the target, and a list of numpy arrays
        for the surrogates. 
    """
    # Extract the time window from the Target as an array:
    df_wnd = df_data[df_data.timestamp.between(window[0], window[1])].copy()
    if verbose: print('Window DF shape (rows, cols) =', df_wnd.shape) 
    if is_check and interval is not None:
        check_window_boundaries(df_wnd, window, 'Arrays', interval, verbose)

    # Get the station names from the dataframe columns:
    target_station = df_wnd.columns[1]
    surr_stations = list(df_wnd.columns[2:])
    if 'Regression' in surr_stations: surr_stations.remove('Regression')
    if 'Regression_SD' in surr_stations: surr_stations.remove('Regression_SD')
    if 'Day_sin' in surr_stations: surr_stations.remove('Day_sin')
    if 'Day_cos' in surr_stations: surr_stations.remove('Day_cos')
    if 'Year_sin' in surr_stations: surr_stations.remove('Year_sin')
    if 'Year_cos' in surr_stations: surr_stations.remove('Year_cos')

    # Get Target as an array:
    target = df_wnd[target_station].to_numpy()

    # Get Regression as an array:
    if is_reg:
        reg = df_wnd.Regression.to_numpy()
        std = df_wnd.Regression_SD.to_numpy()
        day_sin = df_wnd.Day_sin.to_numpy()
        day_cos = df_wnd.Day_cos.to_numpy()
        year_sin = df_wnd.Year_sin.to_numpy()
        year_cos = df_wnd.Year_cos.to_numpy()
        
    # Get a list of Surrogate arrays:
    surrogates = []
    for station in surr_stations:
        surrogates.append(df_wnd[station].to_numpy())

    if is_reg:
        return target, surrogates, reg, std, day_sin, day_cos, year_sin, year_cos
    else:
        return target, surrogates

#-------------------------------------------------------------------------------
# derive_interval
#-------------------------------------------------------------------------------
def derive_interval(df: pd.DataFrame, verbose: int=1) -> int:
    """
    Derives the time interval (minutes) between sensor readings.

    :param df: A dataframe with a 'timestamp' column.
    :return: Returns the interval, in minutes.
    """
    time_diffs = df.timestamp - df.timestamp.shift()
    interval = round(Counter(time_diffs).most_common()[0][0].total_seconds() / 60)
    if verbose: print(f'Measured interval to be {interval} minutes.')
    return interval

#-------------------------------------------------------------------------------
# read_stations
#-------------------------------------------------------------------------------
def read_stations(target_station: str, surrogate_stations: List[str],
    path_dir_data: str, parameter: str, verbose: int=1) \
    -> Tuple[pd.DataFrame, dict, int]:
    """
    Reads dataframes as CSV for the target and surrogates.

    :param target_station: Station name of the target series to impute.
    :param surrogate_stations: List of names of surrogate stations.
    :param path_dir_data: Path to the directory containing subdirectories for
        each parameter .
    :param parameter: Type of sensor, such as GH, WT, PH, DO, SC.
    :return: Returns a dataframe for the target, a dictionary for the surrogates
        where the keys are the station names and the values are the dataframes,
        and the interval between sensor readings in minutes.
    """
    # Derive sub-directory of station data for this particular parameter:
    path_dir_param = join(path_dir_data, parameter)

    # Read target:
    path_station = join(path_dir_param, 'correcteddf.csv')                                                         
    df_target = pd.read_csv(path_station, parse_dates=[0]) \
        .rename(columns={'Unnamed: 0': 'timestamp'})
    df_target.drop(['Raw'], axis=1, inplace=True)
    if verbose:
        print('Target DF shape (rows, cols) =', df_target.shape)
        #print(df_target.head())
        #print(df_target.info())

    # Derive the time interval (minutes) between sensor readings:
    interval = derive_interval(df_target)
    if verbose: print(f'Measured target interval to be {interval} minutes.')

    # Read surrogates:
    dict_surr = {}
    for station in surrogate_stations:

        # Read file:
        path_surr = join(path_dir_param, 'correcteddf.csv')                                                    
        df_surr = pd.read_csv(path_surr, parse_dates=[0]) \
            .rename(columns={'Unnamed: 0': 'timestamp'})
        df_surr.drop(['Raw'], axis=1, inplace=True)

        # Add this station's DF to dictionary:
        dict_surr[station] = df_surr

    return df_target, dict_surr, interval

#-------------------------------------------------------------------------------
# create_resampled_df
#-------------------------------------------------------------------------------
def create_resampled_df(target_station: str, df_target: pd.DataFrame,
    dict_surr: dict, period: List[datetime], interval: int=None,
    verbose: int=1) -> Tuple[pd.DataFrame, int]:
    """
    Creates a dataframe of resampled series such that there is a 'timestamp'
    column and a column for the target and each surrogate where the column names
    are the station names.

    :param target_station: Station name of the target series to impute.
    :param df_target: Dataframe of the target station.
    :param dict_surr: Dictionary of surrogates where the keys are the station
        names and the values are the dataframes.
    :param period: The list of timestamps demarking the period to extract.
    :param interval: Time between sensor readings, in minutes.
    :return: Returns the dataframe and the interval between sensor readings
        in minutes.
    """
    # Extract period and check boundaries:
    df_wnd = extract_window(df_target, period, 'Target', interval, verbose=1)

    # Derive the time interval (minutes) between sensor readings:
    if interval == None:
        interval = derive_interval(df_wnd)
    if verbose: print(f'Measured target interval to be {interval} minutes.')

    # Insert any missing timepoints and impute any gaps:
    df_target = resample_series(df_wnd, interval)
    if verbose: print('Resampled Target DF shape (rows, cols) =', df_target.shape)
    period_pnts = len(df_target)
    if verbose: print(f'Inverval is {interval} minutes. Period includes {period_pnts} points.')
    # Debug: df_target.plot('timestamp', 'Corrected')

    # Create output DF:
    df_data = df_target.rename(columns={'Corrected': target_station})

    # Foreach surrogate station:
    stations = list(dict_surr.keys())
    for station in stations:

        # Access this station's DF:
        df_surr = dict_surr[station]

        # Extract period and check boundaries:
        df_wnd = extract_window(df_surr, period, 'Surrogate' + station, interval)

        # Insert any missing timepoints and impute any gaps:
        df_surr = resample_series(df_wnd, interval)
        assert (len(df_surr) == period_pnts), 'Surrogate missing data points!'
        # Debug: df_surr.plot('timestamp', 'Corrected')

        # Add column to output DF:
        df_data[station] = df_surr.Corrected

    if verbose: print(df_data.head())
    return df_data, interval

#-------------------------------------------------------------------------------
# create_array_datasets
#-------------------------------------------------------------------------------
def create_array_datasets(df_data: pd.DataFrame, interval: int,
    gap: List[datetime], nn_samples: int, gap_pnts: int, is_reg_input: bool,
    is_rhythm: bool=True, verbose: int=1) \
    -> Tuple[np.array, np.array, np.array, np.array, np.array]:
    """
    This method gathers data to use for training a neural network and linear
    regression.  The network's training data contains the following channels:
      - Previous sample of target series
      - Next sample of target series
      - Imputation sample's daily and yearly sine waves
      - Imputation sample's surrogate series
      - Imputation sample's linear regression output

    :param df_data: the dataframe of target and surrogates.
    :param interval: Time between sensor readings, in minutes.
    :param gap: List of timestamps demarking the gap's endpoints.
    :param nn_samples: Number of samples to use for training/validating network.
    :param gap_pnts: Number of data points across the gap.
    :param is_reg_input: Whether to use regression's output as a model input.
    :param is_rhythm: Whether add input channels for diurnal rhythms.
    :return: Returns numpy arrays for training, test, and regression.
    """
    pre_samples = 1
    post_samples = 1
    samples_before_gap = pre_samples + nn_samples + post_samples

    # Define the time window to process:
    reg_samples = 1 if nn_samples == 0 else 0
    window, window_samples = derive_period(interval, gap, nn_samples,
        reg_samples)
    num_samples = nn_samples
    if verbose: print(f'Window of {window_samples} samples for imputation of {num_samples} samples.')

    # Extract the period from the dataframe as an array:
    target, surrogates, reg, std, day_sin, day_cos, year_sin, year_cos = \
        extract_window_as_arrays(df_data, window, True, interval)

    # Handle the special case of linear regression only:
    if nn_samples == 0:
        num_surr = len(surrogates)
        x_test = np.zeros((num_surr, gap_pnts))
        for surr in range(num_surr):
            i1 = gap_pnts
            i2 = i1 + gap_pnts
            x_test[surr] = surrogates[surr][i1:i2]
        return target, None, x_test, std, reg

    # Allocate numpy arrays for CNN inputs:
    #   x = (samples, channels, features)
    #   y = (samples, features)
    num_surr = len(surrogates)
    num_channels_base = 2 # Previous sample, Next sample.
    if is_rhythm:
        num_channels_base += 4 # Day sin & cos, Year sin & cos.
    num_channels = num_channels_base + num_surr
    if is_reg_input: num_channels += 1
    x_train = np.zeros((num_samples, num_channels, gap_pnts))
    x_test = np.zeros((1, num_channels, gap_pnts))
    y_train = np.zeros((num_samples, gap_pnts))
    y_test = np.zeros((1, gap_pnts))
    y_reg = np.zeros((1, gap_pnts))

    # Generate training samples:
    for sample in range(num_samples):

        # Derive indices into the data for the day boundaries:
        i1 = sample * gap_pnts             # Start pre sample.
        i2 = i1 + pre_samples * gap_pnts   # Start gap.
        i3 = i2 + gap_pnts                 # End gap, start post sample.
        i4 = i3 + post_samples * gap_pnts  # End post sample.
        #print(sample, i1, i2, i3, i4)

        # Target on missing sample:
        y_train[sample, :] = target[i2:i3]

        # Target on prior sample:
        x_train[sample, 0, :] = target[i1:i2]

        # Target on post sample:
        x_train[sample, 1, :] = target[i3:i4]

        # Diurnal rhythms:
        if is_rhythm:
            x_train[sample, 2, :] = day_sin[i3:i4]
            x_train[sample, 3, :] = day_cos[i3:i4]
            x_train[sample, 4, :] = year_sin[i3:i4]
            x_train[sample, 5, :] = year_cos[i3:i4]

        # Surrogates on missing sample:
        for surr in range(num_surr):
            x_train[sample, num_channels_base + surr, :] = surrogates[surr][i2:i3]

        # Regression on missing sample:
        if is_reg_input:
            x_train[sample, num_channels_base + num_surr, :] = reg[i2:i3]
 
    # Generate 1 test sample for predicting the gap:
    sample = 0

    # Derive indices into the data for the day boundaries:
    i1 = (samples_before_gap - pre_samples) * gap_pnts # Start pre sample.
    i2 = i1 + pre_samples * gap_pnts                   # Start gap.
    i3 = i2 + gap_pnts                                 # Start post sample.
    i4 = i3 + post_samples * gap_pnts                  # End post sample.

    # Target on missing day:
    y_test[sample, :] = target[i2:i3]

    # Regression on missing day:
    y_reg[sample, :] = reg[i2:i3]

    # Target on prior day:
    x_test[sample, 0, :] = target[i1:i2]

    # Target on post day:
    x_test[sample, 1, :] = target[i3:i4]

    # Diurnal rhythms:
    if is_rhythm:
        x_test[sample, 2, :] = day_sin[i3:i4]
        x_test[sample, 3, :] = day_cos[i3:i4]
        x_test[sample, 4, :] = year_sin[i3:i4]
        x_test[sample, 5, :] = year_cos[i3:i4]

    # Surrogates on missing day:
    for surr in range(num_surr):
        x_test[sample, num_channels_base + surr, :] = surrogates[surr][i2:i3]

    # Regression on missing day:
    if is_reg_input:
        x_test[sample, num_channels_base + num_surr, :] = reg[i2:i3]

    return x_train, y_train, x_test, y_test, y_reg

#------------------------------------------------------------------------------
# create_tf_datasets
#------------------------------------------------------------------------------
def create_tf_datasets(x_in: np.array, y_in: np.array, x_test: np.array,
    y_test: np.array, is_per_channel: bool=False, verbose: int=1) -> tuple:
    """
    Creates TensorFlow datasets (tf.data.Dataset) from numpy arrays.

    :param x_in, y_in: The arrays of training and validation data.
    :param x_test, y_test: The arrays of test data (to impute).
    :param is_per_channel: Whether to normalize each channel independently.
        NOTE: Match this up with the argument to aqi_train.predict().
    :return: Returns datasets for training, validation, and test.
    """
    if verbose: print('TF:', tf.__version__)

    # Derive size of network input and output:
    num_samples, num_channels, num_in = x_in.shape
    num_out = y_in.shape[1]
    if verbose:
        print('Train/Val X shape:', x_in.shape, ", Y shape:", y_in.shape)
        print('Test X shape:', x_test.shape, ", Y shape:", y_test.shape)

    # Shuffle and split data (80%, 20%) for training and validation.
    # Make sure the 3 most recent days always lie in the training set.
    num_recent = 3
    recent_pivot = num_samples - num_recent
    pivot = int(num_samples * 0.8) - num_recent
    idx = np.arange(num_samples)
    idx_old = idx[0:recent_pivot] # The first samples are the oldest.
    idx_recent = idx[recent_pivot:] # The last samples are the newest.
    random.shuffle(idx_old)
    idx_train = np.concatenate((idx_old[0:pivot], idx_recent))
    idx_val = idx_old[pivot:]
    x_train = x_in[idx_train]
    y_train = y_in[idx_train]
    x_val   = x_in[idx_val]
    y_val   = y_in[idx_val]
    if verbose: print(f'Channels: {num_channels}, Features: {num_in}, Train: {len(x_train)}, Val: {len(x_val)}')

    # Set to channels-last (samples, timepoints, channels):
    x_train = np.swapaxes(x_train, 1, 2)
    x_val   = np.swapaxes(x_val,   1, 2)
    x_test  = np.swapaxes(x_test,  1, 2)
    if verbose:
        print('X Train shape:', x_train.shape, ', Val shape:', x_val.shape, ', Test shape:', x_test.shape)
        print('Y Train shape:', y_train.shape, ', Val shape:', y_val.shape, ', Test shape:', y_test.shape)

    # Normalization: 
    #  - Subtract the mean and divide by the standard deviation of each feature.
    #  - The mean and standard deviation should only be computed using the training
    #    data so that the models have no access to the values in the validation set.
    norm = measure_norm_stats(x_train, is_per_channel)
    x_train = normalize(x_train, norm, is_per_channel)
    y_train = normalize(y_train, norm, is_per_channel)
    x_val   = normalize(x_val, norm, is_per_channel)
    y_val   = normalize(y_val, norm, is_per_channel)
    x_test  = normalize(x_test, norm, is_per_channel)
    y_test  = normalize(y_test, norm, is_per_channel)

    # Create datasets:
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_dataset   = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    test_dataset  = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    num_train = len(train_dataset)
    if verbose:
        print('Train dataset of length', num_train)
        print(train_dataset)

    # Shuffle and batch the datasets:
    batch_size = 4
    train_dataset = train_dataset.shuffle(num_train).batch(batch_size)
    val_dataset = val_dataset.batch(batch_size)
    test_dataset = test_dataset.batch(1)
    if verbose:
        print('Train dataset after batching of length', len(train_dataset))
        print(train_dataset)

    return num_in, num_out, num_channels, norm, train_dataset, val_dataset, test_dataset

#-------------------------------------------------------------------------------
# __main__
#-------------------------------------------------------------------------------
if __name__ == "__main__":
    """
    This function is for unit testing of the methods in this script.
    """
    print("aqi_data")
    start_time = time.time()

    # Read parameters from YAML file:
    with open ('aqi_parameters_test_gap.yml', 'r') as stream:
        config = yaml.safe_load(stream)
    path_dir_data = config['path_dir_data']
    parameter = config['parameter']

    # Show parameters:
    print('Parameter:', parameter, 'path_dir_data:', path_dir_data)

    # Unit tests:
    # TODO

    # Report elapsed time:
    secs = time.time() - start_time
    mins = int(secs / 60)
    secs = int(secs - mins * 60)
    print('Elapsed time: %0d minutes, %02d seconds.' % (mins, secs))