#------------------------------------------------------------------------------
# Title: aqi_impute_series.py
# Copyright: Danaher Digital
# Time: 2021
# Desc: This script imputes an entire series, one day at a time.
#------------------------------------------------------------------------------
from os.path import join
import numpy as np
import pandas as pd
from typing import List, Tuple
from scipy.stats import norm
import matplotlib.pyplot as plt
import time
import yaml
import logging
import warnings
from datetime import datetime, timedelta
from aqi_utils import compute_metrics
from aqi_data import extract_window
from aqi_impute_gap import read_stations, create_resampled_df, impute_gap
warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(level=logging.INFO) # Change INFO to DEBUG to diagnose.

#-------------------------------------------------------------------------------
# sample_pdf
#-------------------------------------------------------------------------------
def sample_pdf(pdf: np.array, start: int=0) -> int:
    """
    Generates a random sample from a PDF.

    :param pdf: array of bins representing the PDF like a histogram.
    :param start: first bin to use when sampling.
    :return: Returns the index of the bin that was chosen at random.
    """
    # Compute PDF properties:
    n = len(pdf)      # Number of bins.
    s = int(sum(pdf)) # Sum over all bins.

    # Sample the PDF:
    i = np.random.randint(0, s)
    for j in range(start, n):
        i -= int(pdf[j])
        if i <= 0: break
    return j

#-------------------------------------------------------------------------------
# sample_gap_pdf
#-------------------------------------------------------------------------------
def sample_gap_pdf(gap_type: str) -> int:
    """
    Generates a random sample from the specified type of distribution for gap
    duration.
    
    :param gap_type: one of {'Fixed', 'Uniform', 'Piecewise', 'Short_MoG', 'Long_MoG'}
    """
    # Sample the distribution of gap sizes:
    if gap_type == 'Fixed':
        gap_hours = 24

    elif gap_type == 'Uniform':
        gap_hours = np.random.randint(1, 48 + 1) # Answer lies on [1, 48].

    elif gap_type == 'Piecewise':
        # Construct a PDF as a piecewise-linear model:
        a = np.linspace(12, 16, 3) # Ramp up
        b = np.array([16]) # Flat ceiling
        c = np.linspace(16, 2, 15) # Ramp down
        d = np.ones([29]) * 2 # Flat floor
        y = np.concatenate((a, b, c, d)).astype(int)
        gap_hours = 1 + sample_pdf(y) # Answer lies on [1, 48].

    elif gap_type == 'Short_MoG':
        # Construct a PDF as a mixture of Gaussians:
        sd = 2.5
        x = np.linspace(0, 30, 30)
        a = 100 * norm.pdf(x, 5, sd)
        b = 100 * norm.pdf(x, 24, sd)
        y = np.rint(a + b).astype(int)
        y[0:4] = 0 # Brian said not to test gaps under 4 hours.
        gap_hours = sample_pdf(y, 4) # Answer lies on [4, 29]

    elif gap_type == 'Long_MoG':
        # Construct a PDF as a mixture of Gaussians:
        sd = 3.5
        x = np.linspace(0, 730, 730)
        a = 100 * norm.pdf(x, 7 * 24, sd)
        b = 100 * norm.pdf(x, 30 * 24, sd)
        y = np.rint(a + b).astype(int)
        gap_hours = sample_pdf(y)

    else:
        exit('!!! ERROR unsupported gap type')

    return gap_hours

#-------------------------------------------------------------------------------
# draw_gap
#-------------------------------------------------------------------------------
def draw_gap(impute: List[datetime], interval: int, gap_start: datetime,
    gap_type: str, verbose: int=1) -> Tuple[List[datetime], int, datetime]:
    """
    Draws a gap at random that lies entirely within the series being imputated.

    :param impute: List of timestamps demarking the series of data to impute.
    :param inteval: Time between sensor readings, in minutes.
    :param gap_start: Timestamp marking the gap's beginning.
    :param gap_type: One of {Fixed, Uniform, Piecewise, Short_MoG, Long_MoG}
    :return: Returns the gap as a list of timestamps demarking its endpoints.
    """
    # Sample the distribution of gap sizes:
    gap_hours = sample_gap_pdf(gap_type)

    # Express the duration of the gap in other units:
    gap_mins = gap_hours * 60
    gap_intervals = gap_mins // interval

    # How far to look into the future, until the last point of the gap, which
    # lies at the beginning of the last interval:
    future_intervals = gap_intervals - 1
    future_secs = future_intervals * interval * 60

    # Look forward to the last point of the gap:
    gap_end = gap_start + timedelta(seconds=future_secs)

    # Check for passing the end of the imputation period:
    if gap_end > impute[1]:
        gap_end = impute[1]
        # Work backwards:
        future_secs = (gap_end - gap_start).total_seconds()
        future_intervals = int(future_secs // (interval * 60))
        gap_intervals = future_intervals + 1
        gap_mins = gap_intervals * interval
        gap_hours = gap_mins / 60

    # Look forward by one more interval to find where to begin the next gap:
    next_start = gap_end + timedelta(seconds=interval * 60)

    gap = [gap_start, gap_end]
    if verbose: print('Hours:', gap_hours, ', Points:', gap_intervals, \
        ', Gap:', gap[0], '-', gap[1], ', Next start:', next_start)
    return gap, gap_intervals, next_start

#------------------------------------------------------------------------------
# load_series
#------------------------------------------------------------------------------
def load_series(target_station: str, surrogate_stations: List[str],
    path_dir_data: str, parameter: str, impute: List[datetime],
    max_days: List[int]=[65, 2], verbose: int=1) \
    -> Tuple[pd.DataFrame, int]:
    """
    Reads the CSV dataframes for the target and surrogate stations.

    :param target_station: Station name of the target.
    :param surrogate_stations: List of surrogate station names.
    :param path_dir_data: Path of the directory containing subdirectories
        for each parameter.
    :param impute: List of timestamps demarking the period to impute.
    :param max_days: List of the maximum days to look backward/forward from gap.
    :return: Returns a dataframe with columns for timestamp, target, and
        each surrogate, and also the interval in minutes.
    """
    pre_limit, post_limit = max_days

    # Derive the 'term' (period large enough for all gaps in the series):
    term = [impute[0] - timedelta(days=pre_limit),
        impute[1] + timedelta(days=post_limit)]
    if verbose: print('Term:', term[0], 'to', term[1])

    # Read station data files:
    df_target, dict_surr, interval = read_stations(target_station,
        surrogate_stations, path_dir_data, parameter, verbose=1)

    # Create a dataframe of resampled series:
    df_data, interval = create_resampled_df(target_station, df_target,
        dict_surr, term, verbose=1)

    return df_data, interval

#------------------------------------------------------------------------------
# impute_series
#------------------------------------------------------------------------------
def impute_series(df_data: pd.DataFrame, interval: int, parameter: str,
    impute: list, gap_type: str, max_days: List[int]=[65, 2],
    minmax_nn_samples: List[int]=[10, 60], desc: str=None,
    is_running_avg: bool=False, verbose: int=1) \
    -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Imputes gaps, strung end-to-end, all the way along a series of data.

    :param df_data: Dataframe with columns for timestamp, target, and each
        surrogate.
    :param interval: Time between sensor reading, in minutes.
    :param parameter: The type of sensor, such as GH, WT, PH, SC, DO.
    :param impute: List of timestamps demarking the endpoints of the
        period of time during which to impute gaps.
    :param gap_type: type of gap, such as short Mixture of Gaussians (MoG).
    :param max_days: List of the maximum days to look backward/forward from gap.
    :param minmax_nn_samples: Min and max samples for train/val neural network.
    :param desc: Optional description of the series.
    :param is_running_avg: Whether to print the running average of the metrics
        after each gap is imputed.
    :return: Returns dataframes for the metrics, series data, and gaps.
    """
    pre_limit, post_limit = max_days

    # fine tune impute values to match the interval in the dataset
    most_common_min = df_data.timestamp.apply(lambda x: x.minute).mode()[0]
    impute[0] = impute[0].replace(minute=most_common_min)
    interval_series = pd.date_range(start=impute[0], end=impute[1], freq=f'{interval}min')
    impute[0] = interval_series[0].to_pydatetime()
    impute[1] = interval_series[-1].to_pydatetime()

    # Derive the 'term' (period large enough for all gaps in the series):
    term = [impute[0] - timedelta(days=pre_limit),
        impute[1] + timedelta(days=post_limit)]
    if verbose == 2: print('Term:', term[0], 'to', term[1])

    # Extract term and check boundaries:
    df_term = extract_window(df_data, term, 'Data', interval, verbose=1)

    # Reset running averages:
    if is_running_avg:
        num_metrics = 4
        sum_pred = np.zeros(num_metrics)
        sum_reg = np.zeros(num_metrics)

    # Allocate the output array of imputed series:
    impute_mins = int((impute[1] - impute[0]).total_seconds() // 60)
    impute_pnts = int(impute_mins // interval) + 1 # +1 to include last pnt.
    series_true = np.zeros(impute_pnts)
    series_pred = np.zeros(impute_pnts)
    series_reg = np.zeros(impute_pnts)
    series_band = np.zeros(impute_pnts)

    # Create gap DF with a row per gap:
    df_gaps = pd.DataFrame(columns=['Number', 'Gap_Start', 'Gap_End', \
        'Gap_Points', 'Gap_Hours', 'NN_Samples', 'Algorithm', 'Pred_r', \
        'Linear_r', 'Pred_MAE', 'Linear_MAE', 'Pred_RMSE', 'Linear_RMSE', \
        'Pred_SMAPE', 'Linear_SMAPE'])

    # Foreach gap to impute:
    next_start = impute[0]
    num_gaps = 0
    num_pnts = 0
    is_done = False
    while not is_done:

        # Draw the next gap at random:
        gap, gap_pnts, next_start = draw_gap(impute, interval, next_start,
            gap_type, verbose=1)
        gap_hours = gap_pnts * interval / 60
        num_gaps += 1
        i1 = num_pnts
        i2 = i1 + gap_pnts
        num_pnts += gap_pnts
        if verbose >= 1:
            print('Gap #', num_gaps, '(', gap_pnts, 'points):', gap[0] , '-', gap[1])

        # Check for reaching the end of the imputation period:
        if gap[1] >= impute[1]:
            is_done = True

        # Impute the gap:
        y_true, y_pred, y_reg, band, metrics_reg, metrics_pred, algo, \
            nn_samples = impute_gap(df_term, interval, parameter, gap, max_days,
            minmax_nn_samples, verbose=1)

        # Insert gap's results into series:
        series_true[i1:i2] = y_true
        series_pred[i1:i2] = y_pred
        series_reg[i1:i2] = y_reg
        series_band[i1:i2] = band

        # Append row to gap DF:
        df_gaps.loc[len(df_gaps.index)] = [num_gaps, gap[0], gap[1], gap_pnts, \
            gap_hours, nn_samples, algo, metrics_pred[0], metrics_reg[0], \
            metrics_pred[1], metrics_reg[1], metrics_pred[2], metrics_reg[2], \
            metrics_pred[3], metrics_reg[3] ]

        # Report running average:
        if is_running_avg:
            sum_pred += metrics_pred
            sum_reg += metrics_reg
            avg_pred = sum_pred / num_gaps
            avg_reg = sum_reg / num_gaps
            if verbose >= 1:
                print('r = (%.4f, %.4f), MAE = (%.4f, %.4f), RMSE = (%.4f, %.4f), SMAPE = (%.4f, %.4f)' % \
                    (avg_pred[0], avg_reg[0], avg_pred[1], avg_reg[1],
                    avg_pred[2], avg_reg[2], avg_pred[3], avg_reg[3]) )

    # Compute metrics for both prediction and regression:
    metrics_pred = compute_metrics(series_true, series_pred)
    metrics_reg = compute_metrics(series_true, series_reg)
    if verbose >= 1:
        print(f'{desc} results (Prediction, Regression):')
        print('r = (%.4f, %.4f), MAE = (%.4f, %.4f), RMSE = (%.4f, %.4f), SMAPE = (%.4f, %.4f)' % \
            (metrics_pred[0], metrics_reg[0], metrics_pred[1], metrics_reg[1],
            metrics_pred[2], metrics_reg[2], metrics_pred[3], metrics_reg[3]) )
    
    # Create metrics DataFrame:
    dict_metrics = {'Pred': metrics_pred, 'Reg': metrics_reg}
    df_metrics = pd.DataFrame(dict_metrics)
    if verbose == 2:
        print('Metrics DF shape (rows, cols) =', df_metrics.shape)
        logging.debug(df_metrics.head())

    # Create series DataFrame:
    df_series = df_term[df_term.timestamp.between(impute[0], impute[1])].copy()
    df_series.loc[:, 'Predicted'] = series_pred
    df_series.loc[:, 'Confidence'] = series_band
    if verbose == 2:
        print('Series DF shape (rows, cols) =', df_series.shape)
        print(df_series.head())

    # Plot a chart:
    #TODO plot_prediction(x_pred[0], true, pred, reg, band)

    return df_metrics, df_series, df_gaps

#-------------------------------------------------------------------------------
# unit_test_pdf
#-------------------------------------------------------------------------------
def unit_test_pdf():

    # Test parameters:
    gap_type = 'Short_MoG' # One of {Fixed, Uniform, Piecewise, Short_MoG, Long_MoG}
    hours = 30 # Use 48 for Piecewise, 30 for Short_MoG, 730 for Long_MoG.
    n = 10000

    # Sample the distribution 'n' times:
    y = np.zeros(n)
    for i in range(n):
        y[i] = sample_gap_pdf(gap_type)

    # Plot the histogram:
    plt.hist(y, bins=np.linspace(0, hours))
    plt.show()

#-------------------------------------------------------------------------------
# __main__
#-------------------------------------------------------------------------------
if __name__ == "__main__":
    """
    This function is for unit testing of the methods in this script.
    """
    print("aqi_impute_series")
    start_time = time.time()

    # Read parameters from YAML file:
    with open ('aqi_parameters_test_series.yml', 'r') as stream:
        config = yaml.safe_load(stream)
    path_dir_data = config['path_dir_data']
    max_days = config['max_days']
    gap_type = config['gap_type']
    time_format = config['time_format']
    impute = [datetime.strptime(config['impute'][0], time_format),
        datetime.strptime(config['impute'][1], time_format)]
    parameter = config['parameter']
    target_station = config['target_station']
    surrogate_stations = config['surrogate_stations']

    # Show parameters:
    print('Parameter:', parameter, ', Impute:', impute[0], '-', impute[1],
        ', Target:', target_station)

    # Unit tests:
    # Test: unit_test_pdf()

    # An alternative to calling this method is to read a feather file:
    df_data, interval = load_series(target_station, surrogate_stations,
        path_dir_data, parameter, impute, max_days, verbose=1)

    # Impute gaps strung end-to-end along the series:
    desc = f'Station {target_station}'
    df_metrics, df_series, df_gaps, = impute_series(df_data, interval,
        parameter, impute, gap_type, max_days, desc=desc, verbose=1)

    # Write results:
    name = parameter + '_Metrics_' + target_station + '.csv'
    path_metrics = join(path_dir_data, name)
    df_metrics.to_csv(path_metrics, header=True, index=False, float_format='%.4f')
    #
    name = parameter + '_Series_' + target_station + '.csv'
    path_series = join(path_dir_data, name)
    df_series.to_csv(path_series, header=True, index=False)
    #
    name = parameter + '_Gaps_' + target_station + '.csv'
    path_gaps = join(path_dir_data, name)
    df_gaps.to_csv(path_gaps, header=True, index=False)

    # Report elapsed time:
    secs = time.time() - start_time
    mins = int(secs / 60)
    secs = int(secs - mins * 60)
    print('Elapsed time: %0d minutes, %02d seconds.' % (mins, secs))