#-------------------------------------------------------------------------------
# Title: aqi_impute_gap.py
# Copyright: Danaher Digital
# Time: 2021
# Desc: This script imputes a single gap in time-series data.
#-------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import time
import yaml
from typing import List, Tuple
import logging
import warnings
from datetime import datetime
from aqi_utils import compute_metrics
from aqi_data import derive_nn_samples, derive_period, read_stations, \
    create_resampled_df, create_array_datasets, create_tf_datasets
from aqi_model import build_model
from aqi_train import regress_series, train, predict, plot_prediction, \
    calc_confidence_band, linear_boundary_correction
warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(level=logging.INFO) # Change INFO to DEBUG to diagnose.

#-------------------------------------------------------------------------------
# judge_algorithms
#-------------------------------------------------------------------------------
def judge_algorithms(y_true: np.array, y_linear: np.array, y_pred: np.array,
    percent_r: float=120, percent_error: float=60, num_error: int=3,
    epsilon: float=1e-5) -> bool:
    """
    Judges which algorithm performed better on validation data.

    Returns whether Linear regression is better than the CNN.
    """
    num_samples = len(y_true)
    num_metrics = 4

    # Average over all samples:
    avg_linear = np.zeros(num_metrics)
    avg_pred = np.zeros(num_metrics)
    for i in range(num_samples):
        metrics_linear = compute_metrics(y_true[i], y_linear[i])
        metrics_pred = compute_metrics(y_true[i], y_pred[i])
        avg_linear += metrics_linear
        avg_pred += metrics_pred
    avg_linear /= num_samples
    avg_pred /= num_samples

    # Avoid division by 0:
    if np.any(abs(avg_linear) <= epsilon):
        return False

    # Calculate percentage improvement for each metric:
    r = ((1.0 - avg_linear[0]) - (1.0 - avg_pred[0])) / (1.0 - avg_linear[0]) * 100.0
    mae = (avg_linear[1] - avg_pred[1]) / avg_linear[1] * 100.0
    rmse = (avg_linear[2] - avg_pred[2]) / avg_linear[2] * 100.0
    smape = (avg_linear[3] - avg_pred[3]) / avg_linear[3] * 100.0

    # Decide:
    num_bad = 0
    if mae <= -percent_error: num_bad += 1
    if rmse <= -percent_error: num_bad += 1
    if smape <= -percent_error: num_bad += 1
    if r <= -percent_r and num_bad >= num_error:
        return True
    return False

#-------------------------------------------------------------------------------
# impute_gap
#-------------------------------------------------------------------------------
def impute_gap(df_data: pd.DataFrame, interval: int, parameter: str,
    gap: List[datetime], max_days: List[int]=[65, 2],
    minmax_nn_samples: List[int]=[10, 60], reg_samples: int=3,
    batch_size: int=4, is_reg_input: bool=True, is_per_channel: bool=False,
    verbose: int=1) \
    -> Tuple[np.array, np.array, np.array, np.array, np.array, np.array, str, int]:
    """
    Imputes one gap of time-series data.

    :param df_data: The dataframe with columns for timestamps, target, and
        each surrogate.
    :param interval: Time between sensor readings, in minutes.
    :param parameter: Type of sensor, such as GH, WT, PH, DO, SC.
    :param gap: List of timestamps demarking the gap's endpoints.
    :param max_days: List of the maximum days to look backward/forward from gap.
    :param minmax_nn_samples: Min and max samples for training neural network.
    :param reg_samples: Number of samples to use for training linear regression.
    :param batch_size: Number of samples to train on before updating weights.
    :param is_reg_input: Whether to use regression's output as model input.
    :param is_per_channel: Whether to normalize each channel independently.
    :return: Returns the ground-truth, prediction, regression, and confidence
        during the gap, as well as arrays of metrics for regression and
        prediction, the name of the algorithm used as one of {'CNN', 'Linear'},
        and the number of training samples used.
    """
    # Measure length of gap:
    df_gap = df_data[df_data.timestamp.between(gap[0], gap[1])].copy()
    gap_pnts = len(df_gap)

    # Derive the number of training samples we can use for this gap:
    nn_samples = derive_nn_samples(interval, gap, max_days, minmax_nn_samples,
        reg_samples, verbose=1)

    # Predict every day by linear regression:
    df_data = regress_series(df_data, interval, gap, gap_pnts, nn_samples,
        reg_samples)
    if verbose >= 1: print('Columns:', list(df_data.columns))
    # Debug: df_data.plot('timestamp', [df_data.columns[1], 'Regression'])

    # Gather data into arrays:
    x_train, y_train, x_test, y_test, y_test_reg = create_array_datasets( \
        df_data, interval, gap, nn_samples, gap_pnts, is_reg_input)
    if verbose >= 1:
        if x_train is not None and y_train is not None:
            print('Train shape X:', x_train.shape, ', Y:', y_train.shape)
        if x_test is not None and y_test is not None:
            print('Test shape X:', x_test.shape, ', Y:', y_test.shape)
        print('Reg shape Y:', y_test_reg.shape)

    # Special case of linear regression only:
    if nn_samples == 0:
        is_linear_better = True
        
        # NOTE: create_array_datasets returns (target, None, x_test, std, reg).

        x_pre  = x_train[0:gap_pnts] # The sample before the gap.
        x_post = x_train[2*gap_pnts:] # The points after the gap.
        y_true = x_train[gap_pnts:2*gap_pnts] # In the gap.
        y_pred = y_test_reg[gap_pnts:2*gap_pnts] # In the gap.
        band = y_test[gap_pnts:2*gap_pnts] # In the gap.
        y_pred, offset = linear_boundary_correction(x_pre, y_pred, x_post)

        # Plot the 1st batch as a chart:
        if verbose == 2:
            plot_prediction(x_test, y_true, y_pred, y_pred, band)

        # Compute metrics for both prediction and regression:
        metrics_reg = compute_metrics(y_true, y_pred)
        metrics_pred = metrics_reg
        if verbose >= 1:
            print('r = (%.4f, %.4f), MAE = (%.4f, %.4f), RMSE = (%.4f, %.4f), SMAPE = (%.4f, %.4f)' % \
                (metrics_pred[0], metrics_reg[0], metrics_pred[1], metrics_reg[1],
                metrics_pred[2], metrics_reg[2], metrics_pred[3], metrics_reg[3]) )
        algo = 'Linear'
        return y_true, y_pred, y_pred, band, metrics_reg, metrics_pred, algo, \
            nn_samples

    # Create TensorFlow datasets:
    num_in, num_out, num_channels, norm, train_dataset, val_dataset, \
        test_dataset = create_tf_datasets(x_train, y_train, x_test, y_test,
        is_per_channel)

    # Determine how many levels to use in the U-net:
    # (NOTE: if the number of points in the gap is not a multiple of 4, then we
    # can't use 3 levels because downsampling twice and upsampling twice will
    # not recover the original dimension. Similarly, we can't use more than 1
    # level with a gap that's an odd number of points long.)
    num_levels = int(gap_pnts // 4)
    if num_levels < 1: num_levels = 1
    if num_levels > 3: num_levels = 3 # 4 is too deep for this little data.
    if num_levels == 3 and num_in % 4 > 0:
        num_levels = 2
    if num_levels > 1 and num_in % 2 > 0:
        num_levels = 1

    # Build model:
    model = build_model("UNet", num_in, num_out, num_channels, num_filters=16,
        num_levels=num_levels, is_cat=True, is_res=False, is_bn=False, drop=0,
        is_thick_encoder=False, is_long_bottom=False, is_smooth=True,
        is_tran=False)

    # Train model:
    model, history = train(model, train_dataset, val_dataset,
        batch_size=batch_size, verbose_fit=0)

    # Predict with model (note that outputs have a batch dimension):
    x_pred, y_true, y_pred, y_reg = predict(model, test_dataset, norm, gap_pnts,
        is_reg_input, parameter, is_per_channel)

    # Generate confidence band:
    x_pred_val, y_true_val, y_pred_val, y_reg_val = predict(model, val_dataset,
        norm, gap_pnts, is_reg_input, parameter, is_per_channel)
    band = calc_confidence_band(y_true_val, y_pred_val)

    # Use the algorithm which performed best on validation data:
    is_linear_better = False
    if is_reg_input:
        is_linear_better = judge_algorithms(y_true, y_reg, y_pred)
    if is_linear_better:
        algo = 'Linear'
        y_pred = y_reg
    else:
        algo = 'CNN'

    # Plot the 1st batch as a chart:
    if verbose == 2:
        plot_prediction(x_pred[0], y_true[0], y_pred[0], y_test_reg[0], band)

    # Compute metrics for both prediction and regression:
    metrics_reg = compute_metrics(y_true[0], y_test_reg[0])
    metrics_pred = compute_metrics(y_true[0], y_pred[0])
    if verbose >= 1:
        print('r = (%.4f, %.4f), MAE = (%.4f, %.4f), RMSE = (%.4f, %.4f), SMAPE = (%.4f, %.4f)' % \
            (metrics_pred[0], metrics_reg[0], metrics_pred[1], metrics_reg[1],
            metrics_pred[2], metrics_reg[2], metrics_pred[3], metrics_reg[3]) )

    return y_true[0], y_pred[0], y_test_reg[0], band, metrics_reg, \
        metrics_pred, algo, nn_samples

#-------------------------------------------------------------------------------
# __main__
#-------------------------------------------------------------------------------
if __name__ == "__main__":
    """
    This function is for unit testing of the methods in this script.
    """
    print("aqi_impute_gap")
    start_time = time.time()

    # Read parameters from YAML file:
    with open ('aqi_parameters_test_gap.yml', 'r') as stream:
        config = yaml.safe_load(stream)
    path_dir_data = config['path_dir_data']
    reg_samples = config['reg_samples']
    minmax_nn_samples = config['minmax_nn_samples']
    max_days = config['max_days']
    batch_size = config['batch_size']
    is_reg_input = config['is_reg_input']
    is_per_channel = config['is_per_channel']
    time_format = config['time_format']
    gap = [datetime.strptime(config['gap'][0], time_format),
        datetime.strptime(config['gap'][1], time_format)]
    parameter = config['parameter']
    target_station = config['target_station']
    surrogate_stations = config['surrogate_stations']

    # Show parameters:
    print(parameter, 'Target station:', target_station, ', Surrogates:',
        surrogate_stations)
    print(f'Gap: {gap[0]} - {gap[1]}')

    # Read station data files:
    df_target, dict_surr, interval = read_stations(target_station,
        surrogate_stations, path_dir_data, parameter, verbose=1)

    # Derive the number of CNN training/validation samples to use for this gap:
    nn_samples = derive_nn_samples(interval, gap, max_days,
        minmax_nn_samples, reg_samples, verbose=1)

    # Derive the period of interest, big enough for regression and imputation:
    period, period_samples = derive_period(interval, gap, nn_samples,
        reg_samples, verbose=1)

    # Create a dataframe of resampled series:
    df_data, interval = create_resampled_df(target_station, df_target,
        dict_surr, period, verbose=1)

    # Impute the gap:
    y_true, y_pred, y_reg, band, metrics_reg, metrics_pred, algo, nn_samples = \
        impute_gap(df_data, interval, parameter, gap, max_days, \
        minmax_nn_samples, reg_samples, batch_size, is_reg_input, \
        is_per_channel, verbose=2)
    print('Algorithm:', algo, ', NN samples:', nn_samples)

    # Report elapsed time:
    secs = time.time() - start_time
    mins = int(secs / 60)
    secs = int(secs - mins * 60)
    print('Elapsed time: %0d minutes, %02d seconds.' % (mins, secs))