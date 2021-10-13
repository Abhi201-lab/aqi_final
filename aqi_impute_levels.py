#------------------------------------------------------------------------------
# Title: aqi_impute_levels.py
# Copyright: Danaher Digital
# Time: 2021
# Desc: This script imputes several series, each as several levels of
# surrogate correlation.
#------------------------------------------------------------------------------
from os.path import join
import pandas as pd
import time
from datetime import datetime
import yaml
from aqi_impute_series import load_series, impute_series
import logging
logging.basicConfig(level=logging.INFO) # Change INFO to DEBUG to diagnose.

#-------------------------------------------------------------------------------
# __main__
#-------------------------------------------------------------------------------
if __name__ == "__main__":

    print("aqi_impute_levels")
    start_time = time.time()

    # Read parameters from YAML file:
    with open ('aqi_parameters_test_levels.yml', 'r') as stream:
        config = yaml.safe_load(stream)
    path_dir_data = config['path_dir_data']
    max_days = config['max_days']
    gap_type = config['gap_type']
    time_format = config['time_format']
    impute = [datetime.strptime(config['impute'][0], time_format),
        datetime.strptime(config['impute'][1], time_format)]
    parameter = config['parameter']
    target_stations = config['target_stations']

    # Show parameters:
    print('Parameter:', parameter, ', Impute:', impute[0], '-', impute[1],
        ', Targets:', target_stations)

    # Foreach target station:
    for target_station in target_stations:
    
        # Read LUT which lists surrogate stations for each correlation level:
        name = parameter + '_LUT_' + target_station + '.csv'
        path_lut = join(path_dir_data, name)
        df_lut = pd.read_csv(path_lut)
        df_lut['Stations'] = df_lut['Stations'].apply(eval)
        print('LUT DF shape (rows, cols) =', df_lut.shape)
        logging.debug(df_lut.head())

        # Foreach level of surrogate correlation:
        for idx_level, row in df_lut.iterrows():

            # Read row of LUT:
            level, surrogate_stations = row['R'], row['Stations']
            
            print(f'Processing level #{idx_level} with r = {level} and {len(surrogate_stations)} stations.')

            # An alternative to calling this method is to read a feather file:
            df_data, interval = load_series(target_station, surrogate_stations,
                path_dir_data, parameter, impute, max_days)

            # Impute gaps strung end-to-end along the series:
            desc = f'Station {target_station}, level {level}'
            df_metrics, df_series, df_gaps = impute_series(df_data, interval,
                parameter,impute, gap_type, max_days, desc=desc, verbose=1)

            # Write results:
            name = parameter + '_Metrics_' + target_station + '_' + str(int(level*100)) + '.csv'
            path_metrics = join(path_dir_data, name)
            df_metrics.to_csv(path_metrics, header=True, index=False, float_format='%.4f')
            #
            name = parameter + '_Series_' + target_station + '_' + str(int(level*100)) + '.csv'
            path_series = join(path_dir_data, name)
            df_series.to_csv(path_series, header=True, index=False)
            #
            name = parameter + '_Gaps_' + target_station + '_' + str(int(level*100)) + '.csv'
            path_gaps = join(path_dir_data, name)
            df_gaps.to_csv(path_gaps, header=True, index=False)

    # Report elapsed time:
    secs = time.time() - start_time
    mins = int(secs / 60)
    secs = int(secs - mins * 60)
    print('Elapsed time: %0d minutes, %02d seconds.' % (mins, secs))