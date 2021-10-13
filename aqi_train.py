#------------------------------------------------------------------------------
# Title: aqi_train.py
# Copyright: Danaher Digital
# Time: 2021
# Desc: This script performs model training and prediction.
#------------------------------------------------------------------------------
from os.path import join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import yaml
from datetime import datetime
from typing import List, Tuple
import warnings
import os
from aqi_utils import measure_norm_stats, normalize, unnormalize
from aqi_data import derive_period, derive_interval, extract_window_as_arrays
from sklearn.linear_model import LinearRegression, BayesianRidge
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from tensorflow.keras.utils import plot_model
warnings.simplefilter(action='ignore', category=FutureWarning)

#-------------------------------------------------------------------------------
# linear_boundary_correction
#-------------------------------------------------------------------------------
def linear_boundary_correction(pre: np.array, pred: np.array,
    post: np.array=None, is_constant: bool=False) -> Tuple[np.array, np.array]:
    """
    This method computes a linear correction and applies it to the prediction.
    It uses linear extrapolation of the known data to estimate what the
    boundary points should be, and then calculates a linear ramp across the
    prediction such that the boundary values of the prediction match those of
    the extrapolation.

    :param pre: Known data prior to the target period (1-D time series).
    :param pred: Prediction during the target period (1-D time series).
    :param post: Known data after the target period. (Can be None.)
    :param is_constant: Whether offset should be constant or ramp when no post.
    :return: Returns a new prediction.
    """
    # Check that data exists for extrapolation:
    is_pre = pre is not None and len(pre) > 0
    is_post = post is not None and len(post) > 0
    if not is_pre and not is_post:
        offset = np.zeros(len(pred))
        return pred, offset

    # Boundary values on each side of the prediction (first and last points):
    p1, p2 = pred[0], pred[-1]

    # Perform extrapolations over 2 days and 3 days, and then average them:
    
    # Let 'a' be the projected value of the first day of the prediction:
    v3 = pre[-1]
    v2 = v3 if len(pre) < 2 else pre[-2]
    v1 = v2 if len(pre) < 3 else pre[-3]
    a1 = v3 + (v3 - v2)
    a2 = v3 + (v3 - v1) * 0.5
    a = (a1 + a2) * 0.5

    # Let 'b' be the projected value of the last day of the prediction:
    if is_post:
        v1 = post[0]
        v2 = v1 if len(post) < 2 else post[1]
        v3 = v2 if len(post) < 3 else post[2]
        b1 = v1 - (v2 - v1)
        b2 = v1 - (v3 - v1) * 0.5
        b = (b1 + b2) * 0.5
    
    # Generate the offset as a linear ramp:
    o1 = a - p1
    if is_post:
        o2 = b - p2
    else:
        if is_constant:
            o2 = o1
        else:
            o2 = 0
    offset = np.linspace(o1, o2, num=len(pred))

    # Add the offset:
    pred += offset
    return pred, offset

#-------------------------------------------------------------------------------
# regress_series
#-------------------------------------------------------------------------------
def regress_series(df_data: pd.DataFrame, interval: int, gap: List[datetime],
    gap_pnts: int, nn_samples: int, reg_samples: int,
    regression: str='Bayesian', verbose: int=1) -> pd.DataFrame:
    """
    This method performs linear regression during the gap to be imputed, as well
    as during the training period before and after the gap. The results are
    added to the dataframe as columns "Regression" and "Regression_SD".
    It also adds sine wave columns to represent diurnal rhythms.

    :param df_data: the dataframe of target and surrogates.
    :param interval: Time between sensor readings, in minutes.
    :param gap: List of timestamps demarking the gap's endpoints.
    :param gap_pnts: Number of data points across the gap.
    :param nn_samples: Number of samples to use for training/validating model.
    :param reg_samples: Number of samples to use for training linear regression.
    :param regression: One of {'Bayesian', 'Linear'}.
    :return: Returns the target dataframe with an additional column.
    """
    # Define the time window to process:
    window, window_samples = derive_period(interval, gap, nn_samples,
        reg_samples)
    num_samples = window_samples - reg_samples
    if verbose: print(f'Window of {window_samples} samples for regression of {num_samples} samples.')

    # Locate which rows of the dataframe correspond with the window's
    # timestamps (let 'i1_wnd' be the 0-based index into the dataframe of the
    # first point in the window):
    i1_wnd = int(df_data.index[df_data.timestamp == window[0]].tolist()[0])
    i2_wnd = int(df_data.index[df_data.timestamp == window[1]].tolist()[0])
    i_offset = int(df_data.index[0])
    i1_wnd, i2_wnd = i1_wnd - i_offset, i2_wnd - i_offset

    # Extract the time window from the dataframe as an array:
    target, surrogates = extract_window_as_arrays(df_data, window, is_reg=False)

    # Check if any surrogates exist:
    num_surr = len(surrogates)
    if num_surr == 0: exit('!!! Do not call this method without surrogates !!!')

    # Allocate workspace:
    boundary_pnts = 3
    reg_pnts = reg_samples * gap_pnts
    x_train = np.zeros((num_surr, reg_pnts))
    x_test = np.zeros((num_surr, gap_pnts))
    pred = np.zeros(target.shape)
    std = np.zeros(target.shape)

    # Build the regression model:
    if regression == 'Linear':
        model = LinearRegression()
    elif regression == 'Bayesian':
        model = BayesianRidge(compute_score=True)
    else:
        exit(f'!!! ERROR: unsupported regression model: {regression} !!!')

    # Foreach sample:
    for sample in range(num_samples):
            
        # Localize the data for this sample:
        i1 = sample * gap_pnts  # Start of training.
        i2 = i1 + reg_pnts      # Start gap.
        i3 = i2 + gap_pnts      # End gap.
        i4 = i3 + boundary_pnts # End of points needed for boundary correction.

        # Generate train/test data:
        for surr in range(num_surr):
            x_train[surr, :] = surrogates[surr][i1:i2]
            x_test[surr, :] = surrogates[surr][i2:i3]
        y_train = target[i1:i2]
        y_post = target[i3:i4]

        # Set to channels-last:
        x_train = np.swapaxes(x_train, 0, 1)
        x_test = np.swapaxes(x_test, 0, 1)
       
        # Normalization: 
        #  - Subtract the mean and divide by the standard deviation of each feature.
        #  - The mean and standard deviation should only be computed using the training
        #    data so that the models have no access to the values in the validation set.
        # NOTE: Although least-squares needs no normalization, ridge regression does.
        norm = measure_norm_stats(x_train, False)
        x_train = normalize(x_train, norm, False)
        y_train = normalize(y_train, norm, False)
        x_test  = normalize(x_test, norm, False)
         
        # Train model:
        model.fit(x_train, y_train)

        # Predict using model:
        if regression == 'Bayesian':
            y_pred, y_std = model.predict(x_test, return_std=True)
        else:
            y_pred = model.predict(x_test)
            y_std = np.ones(y_pred.shape)

        # Un-normalize:
        y_train = unnormalize(y_train, norm, False)
        y_pred = unnormalize(y_pred, norm, False)
        y_std = unnormalize(y_std, norm, False)

        # Boundary correction:
        y_pred, offset = linear_boundary_correction(y_train, y_pred, y_post)
    
        # Store result:
        pred[i2:i3] = y_pred
        std[i2:i3] = y_std
        x_train = np.swapaxes(x_train, 0, 1)
        x_test = np.swapaxes(x_test, 0, 1)

        # Prevent there being 0's at the beginning and end of the window:
        if sample == 0:
            pred[i1:i2] = target[i1:i2]
        if sample == num_samples - 1:
            pred[i3:i4] = target[i3:i4]

    # Add columns of results to the dataframe:
    df = df_data.copy() # Wasteful, but avoids "write on a view".
    df.loc[:, 'Regression'] = 0
    df.loc[:, 'Regression_SD'] = 0
    df.loc[df.index[i1_wnd:i2_wnd+1], 'Regression'] = pred
    df.loc[df.index[i1_wnd:i2_wnd+1], 'Regression_SD'] = std

    # Add columns for diurnal rhythms:
    pnts_per_day = 24 * 60 / interval
    pnts_per_year = 365 * pnts_per_day
    df['Day_sin'] = np.sin(df.index * (2 * np.pi / pnts_per_day))
    df['Day_cos'] = np.cos(df.index * (2 * np.pi / pnts_per_day))
    df['Year_sin'] = np.sin(df.index * (2 * np.pi / pnts_per_year))
    df['Year_cos'] = np.cos(df.index * (2 * np.pi / pnts_per_year))

    if verbose:
        print('New DF shape (rows, cols) =', df.shape) 
        print(df.head())
    return df

#------------------------------------------------------------------------------
# train
#------------------------------------------------------------------------------
def train(model, train_dataset, val_dataset, patience: int=5,
    max_epochs: int=200, batch_size: int=4, verbose_fit=0, verbose=1,
    filename: str=None) -> Tuple:
    """
    This method trains the model.

    :param model: The model to train.
    :param train_dataset: The training data.
    :param val_dataset: The validation data, which is used to know when
        to reduce the learning rate and when to stop training to avoid
        overfitting.
    :param patience: The number of epochs to wait for the validation loss to
        decline further.
    :param max_epochs: The maximum number of epochs for which to train.
    :param batch_size: Number of samples in each mini-batch, between updates
        to the weights.
    :param verbose_fit: Whether to print status during training epochs.
    :param verbose: Whether to print other debug messages.
    :param filename: Where to store a diagram of network architecture.
    :return: Returns the trained model and its training history.
    """
    # Callback to reduce learning rate when a metric has stopped improving:
    callback_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
        mode='min', factor=0.5, patience=patience, verbose=verbose_fit,
        min_lr=0.0)
  
    # Callback to stop training when a monitored metric has stopped improving:
    callback_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
        mode='min', patience=int(2.4 * patience), verbose=verbose_fit,
        restore_best_weights=True)

    # Callback that streams epoch results to a CSV file:
    callback_log = tf.keras.callbacks.CSVLogger('training.txt', append=False)

    model.compile(loss=tf.losses.MeanSquaredError(),
        optimizer=tf.optimizers.Adam(), # learning_rate=0.001
        metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(train_dataset, epochs=max_epochs,
        validation_data=val_dataset,
        batch_size=batch_size,
        shuffle=True,
        verbose=verbose_fit,
        callbacks=[callback_lr, callback_stop, callback_log])

    if verbose:
        print(model.summary())

    if filename is not None:
        plot_model(model, to_file=filename + '.png', show_shapes=True,
            show_layer_names=True)

    if verbose:
        val_performance = model.evaluate(val_dataset)
        print('Val performance [loss, mean abs err] =', val_performance)
     
    return model, history

#------------------------------------------------------------------------------
# predict
#------------------------------------------------------------------------------
def predict(model, dataset, norm: Tuple[float], gap_pnts: int,
    is_reg_input: bool, parameter: str=None, is_per_channel: bool=False,
    verbose: int=1) -> Tuple[np.array, np.array, np.array, np.array]:
    """
    This method uses a trained model to perform prediction.
    
    :param model: The trained model.
    :param dataset: The test data.
    :param norm: The stats on training data returned from normalization.
    :param gap_pnts: The number of points across the gap to impute.
    :param is_reg_input: Whether to use regression's output as a model input.
    :param parameter: Type of sensor, such as GH, WT, PH, DO, SC.
    :param is_per_channel: Whether to normalize each channel independently.
        NOTE: Match this up with the argument to aqi_data.create_tf_datasets().
    :return: Returns x, y_true, y_pred, y_reg.
    """
    # Allocate outputs:
    num_batches = len(dataset)
    num_samples = 0
    for x_batch, y_batch, in dataset:
        num_samples += len(x_batch)
        num_pnts, num_channels = x_batch.shape[1], x_batch.shape[2]
    x = np.zeros((num_samples, num_channels, num_pnts))
    true = np.zeros((num_samples, num_pnts))
    pred = np.zeros((num_samples, num_pnts))
    reg = np.zeros((num_samples, num_pnts))

    # Foreach batch:
    prev_samples = 0
    for x_batch, y_batch, in dataset:

        # Predict this batch:
        pred_batch = model(x_batch)

        # Perform boundary-correction of each sample in this batch:
        num_samples = len(x_batch)
        for i_sample in range(num_samples):

            # Extract this sample from the batch:
            x_all, y_true, y_pred = x_batch[i_sample].numpy(), \
                y_batch[i_sample].numpy(), pred_batch[i_sample].numpy()
            if verbose: print('Sample size:', x.shape, y_true.shape)

            # Un-normalize:
            x_all = unnormalize(x_all, norm, is_per_channel)
            y_true = unnormalize(y_true, norm, is_per_channel)
            y_pred = unnormalize(y_pred, norm, is_per_channel)

            # Set to channels-first (channels, timepoints):
            x_all = np.swapaxes(x_all, 0, 1)

            # Correct boundaries:
            x_pre = x_all[0, :] # The sample before the gap.
            x_post = x_all[1, :] # The sample after the gap.
            y_pred, offset = linear_boundary_correction(x_pre, y_pred, x_post)

            # For certain parameters, prevent negative values:
            if parameter in ['DO', 'GH', 'SC', 'TB']:
                y_pred[y_pred < 0] = 0

            # Store results:
            i = prev_samples + i_sample
            x[i] = x_all
            true[i] = y_true
            pred[i] = y_pred
            if is_reg_input:
                reg[i] = x_all[-1]

        # Advance:
        prev_samples += num_samples

    return x, true, pred, reg

#------------------------------------------------------------------------------
# plot_prediction
#------------------------------------------------------------------------------
def plot_prediction(x: np.array, y_true: np.array, y_pred: np.array,
    y_reg: np.array=None, band: np.array=None):
    """
    This method plots a prediction relative to ground truth.
    
    :param x: Multi-channel input data, including surrogates.
    :param y_true: Ground-truth output, single-channel.
    :param y_pred: Predicted outputs, single-channel.
    :param y_reg: Output of Bayesian Ridge linear regression.
    :param band: Confidence of prediction.
    :return: Returns nothing.
    """
    indices = np.arange(len(y_true))
    plt.figure(figsize=(12, 8))
    plt.ylabel('Temperature')
    plt.xlabel('Time')
    plt.title('Prediction vs. Truth')
    for xx in x:
        plt.plot(indices, xx, label='Inputs', marker='.', zorder=-10)

    plt.scatter(indices, y_true, edgecolors='k', label='Labels', c='#2ca02c',
        s=64)
    plt.scatter(indices, y_pred, marker='X', edgecolors='k',
        label='Predictions', c='#ff7f0e', s=64)

    if y_reg is not None:
        plt.scatter(indices, y_reg, marker='*', edgecolors='k',
            label='Regression', c='#7f0eff', s=64)

    if band is not None:
        band_lo = y_pred - band
        band_hi = y_pred + band
        plt.scatter(indices, band_lo, marker='+', edgecolors='k',
            label='band_lo', c='#cccc11', s=64)
        plt.scatter(indices, band_hi, marker='+', edgecolors='k',
            label='band_hi', c='#cccc11', s=64)

    plt.legend()
    plt.show()

#------------------------------------------------------------------------------
# calc_confidence_band
#-----------------------------------------------------------------------------
def calc_confidence_band(y_true: np.array, y_pred: np.array, kpi=0.95,
    verbose: int=1) -> np.array:
    """
    This method calculates the confidence of the prediction, on a per-point
    basis.  The approach is to measure the statistics of the difference
    between the prediction and the truth on the validation data.  Since the
    stats include mean and standard deviation, a total "band" value is
    derived by adding a certain number of standard deviations to the mean.
    This number is determined by measuring what would meet the KPI.
        
    :param y_true: Ground-truth outputs.
    :param y_pred: Predicted outputs.
    :param kpi: Percentage of signal that must lie within band.
    :return: Returns the band.
    """
    # Difference from ground-truth:
    dif = np.abs(y_true - y_pred)
    num_samples, num_pnts = dif.shape

    # Calculate stats of each column:
    stats = np.zeros((num_pnts, 2))
    for i in range(num_pnts):
        i1 = max(i-1, 0)
        i2 = min(i+2, num_pnts)
        col = dif[:, i1:i2]
        m = col.mean()
        sd = col.std()
        stats[i] = m, sd
    if verbose: print(stats)

    # Estimate how many standard deviations are needed to meet KPI:
    search_sd = np.linspace(0.5, 3.0, 6)
    min_count = int(kpi * num_samples * num_pnts)
    for num_sd in search_sd:
        count = 0
        for i in range(num_pnts):
            m, sd = stats[i]
            w = m + num_sd * sd
            for s in range(num_samples):
                pred, true = y_pred[s, i], y_true[s, i]
                if pred - w <= true and true <= pred + w:
                    count += 1
        if count >= min_count:
            break

    # Calculate band:
    band = np.zeros(num_pnts)
    for i in range(num_pnts):
        m, sd = stats[i]
        w = m + num_sd * sd
        band[i] = w

    if verbose:
        print(band)
    return band

#-------------------------------------------------------------------------------
# unit_test_boundary_correction
#-------------------------------------------------------------------------------
def unit_test_boundary_correction():

    pre = np.array([1, 2, 3, 4]).astype(float)
    pred = np.array([6, 7, 7, 6]).astype(float)
    post = np.array([4, 3, 2, 1]).astype(float)
    corr = linear_boundary_correction(pre, pred.copy(), post)
    print('Pred:', pred, ', Corrected:', corr)

#-------------------------------------------------------------------------------
# make_snippet
#-------------------------------------------------------------------------------
def make_snippet(path_dir_param: str, filename_snippet: str, target_station: str):
    """
    Makes a snippet file from a portion of an existing dataframe.
    """
    # Read:
    path_in = join(path_dir_param, target_station + '.csv')
    path_snippet = join(path_dir_param, filename_snippet)
    df_snippet = pd.read_csv(path_in).rename(columns={'Unnamed: 0': 'timestamp'})

    # Snip:
    df_snippet = df_snippet[df_snippet.timestamp >= '2019-07-21']
    df_snippet = df_snippet[df_snippet.timestamp < '2019-07-28']
    print('Snippet DF shape (rows, cols) =', df_snippet.shape)
    print(df_snippet.head())

    # Write:
    df_snippet.to_csv(path_snippet, header=True, index=False)

#-------------------------------------------------------------------------------
# unit_test_sine
#-------------------------------------------------------------------------------
def unit_test_sine(path_dir_data: str, parameter: str):

    # Generate a snippet if one doesn't exist already:
    path_dir_param = join(path_dir_data, parameter)
    # Setup: make_snippet(path_dir_param, '01018035_week', '01018035')

    # Read snippet:
    path_snippet = join(path_dir_param, '01018035_week')
    df = pd.read_csv(path_snippet, parse_dates=[0]) \
        .rename(columns={'Unnamed: 0': 'timestamp'})

    # Generate sine:
    interval = derive_interval(df, verbose=1)
    pnts_per_day = 24 * 60 / interval
    df['Day_sin'] = np.sin(df.index * (2 * np.pi / pnts_per_day)) * 5 + 24
    print(df.head())

    df.plot('timestamp', ['Corrected', 'Day_sin'])
    plt.show()

#-------------------------------------------------------------------------------
# __main__
#-------------------------------------------------------------------------------
if __name__ == "__main__":
    """
    This function is for unit testing of the methods in this script.
    """
    print("aqi_train")
    start_time = time.time()

    # Read parameters from YAML file:
    with open ('aqi_parameters_test_gap.yml', 'r') as stream:
        config = yaml.safe_load(stream)
    path_dir_data = config['path_dir_data']
    parameter = config['parameter']

    # Show parameters:
    print('Parameter:', parameter, 'path_dir_data:', path_dir_data)

    # Unit tests:
    # Test: unit_test_boundary_correction()
    #unit_test_sine(path_dir_data, parameter)

    # Report elapsed time:
    secs = time.time() - start_time
    mins = int(secs / 60)
    secs = int(secs - mins * 60)
    print('Elapsed time: %0d minutes, %02d seconds.' % (mins, secs))