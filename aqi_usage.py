#------------------------------------------------------------------------------
# Title: aqi_usage.py
# Copyright: Danaher Digital
# Time: 2021
# Desc: This script demonstrates example usage of the algorithm API.
#------------------------------------------------------------------------------
from os.path import join
import pandas as pd
import json           
from typing import List
import yaml
from datetime import datetime, timezone
from aqi_data import read_stations, create_resampled_df
from aqi_algo_api import algo_request, algo_impute

#-------------------------------------------------------------------------------
# collect_data
#-------------------------------------------------------------------------------
def collect_data(parameter: str, path_dir_data: str, period: List[datetime],
    target_station: str, surrogate_stations: List[str], interval: int=None, 
    verbose: int=1) -> pd.DataFrame:
    """
    This method reads station data files and extracts the requested time period.
    
    :param parameter: the type of sensor, such as GH, WT, DO, PH, SC, etc.
    :param path_dir_data: the top directory containing parameter subdirectories.
    :param period: list of the 2 timepoints of the period, inclusive.
    :param target_station: station name of the data to impute.
    :param surrogate_stations: list of station names of the surrogates.
    :param interval: Time between sensor readings, in minutes.
    :return: Returns dataframe with columns for timestamps, the target,
        and each surrogate.
    """
    # Read station data files:
    df_target, dict_surr, interval_read = read_stations(target_station,
        surrogate_stations, path_dir_data, parameter, verbose=1)

    # Create a dataframe of resampled series:
    df_data, interval_sampled = create_resampled_df(target_station, df_target,
        dict_surr, period, interval, verbose=1)
    if verbose:
        print('Data DF shape (rows, cols) =', df_data.shape)
        print(df_data.head())

    return df_data    

 
def input_call(parameter,parameterName,locationIdentifier,locationName,target_station,timeRange,surrogate_stations,interval,
                     surrogate_locationId,surrogate_location_name,surrogate_parameterName):

    time_format = '%Y-%m-%dT%H:%M:%S%z'
    parameter = [parameter]
    parameter = str(parameter[0])
    gap1 = [timeRange]
    gap = gap1[0]
    start = gap[0]
    end = gap[1]
    gap = datetime.strptime(start, time_format), datetime.strptime(end, time_format)
    path_dir_data = '/home/ubuntu/Data/AQI/corrected_df'
    target_station = [target_station]
    target_station = ' '.join(map(str, target_station))
    surrogate_stations = [surrogate_stations]
    interval = [interval][0]

    print('Parameter:', parameter, ', Gap:', start, '-', end, ', Target:',
        target_station, ', Surrogates:', surrogate_stations)

    # Read target snippet:
    path_dir_param = join(path_dir_data, parameter)

    # Algorithm call #1 (Request):
    period = algo_request(parameter, gap, interval, verbose=1)

    # Gather data during this period:
    df_data = collect_data(parameter, path_dir_data, period,
        target_station, surrogate_stations, interval, verbose=1)

    # Algorithm call #2 (Impute):
    df_imputed, algo, metrics_reg, metrics_pred = algo_impute(parameter, gap, interval, df_data, verbose=1)
    if algo == 'Linear':
        algorithm = 'linear Bayesian Ridge regression'
        warning = 'Reverted to linear Bayesian Ridge regression'

    else:
        algorithm = 'CNN'
        warning = 'CNN Model'
        
    df_imputed.drop(['Confidence'], axis=1, inplace=True)

    metrics = ('r = (%.4f, %.4f), MAE = (%.4f, %.4f), RMSE = (%.4f, %.4f), SMAPE = (%.4f, %.4f)' % \
            (metrics_pred[0], metrics_reg[0], metrics_pred[1], metrics_reg[1],
            metrics_pred[2], metrics_reg[2], metrics_pred[3], metrics_reg[3]) )

    surrogate_stations = [surrogate_stations][0]
    surrogate_stations = str(surrogate_stations[0])
    surrogate_locationId = [surrogate_locationId]
    surrogate_locationId = str(surrogate_locationId[0])
    surrogate_location_name = [surrogate_location_name]
    surrogate_location_name = str(surrogate_location_name[0])
    surrogate_parameterName = [surrogate_parameterName]
    surrogate_parameterName = str(surrogate_parameterName[0])                                          
    
    TEXT1 = " , "
    TEXT2 = " Surrogates details: "
    TEXT3 = " timeSeriesUniqueId: "
    TEXT4 = " locationIdentifier: "
    TEXT5 = " locationName: "    
    TEXT6 = " parameterName: "    
    
    msg = algorithm + TEXT1 + metrics + TEXT2 + TEXT3 + surrogate_stations + TEXT4 + surrogate_locationId + TEXT5 + surrogate_location_name + TEXT6 + surrogate_parameterName
  

    df_imputed.rename(columns = {'Predicted':'value'}, inplace = True)
    df_imputed = df_imputed.iloc[:, [1,0]]
    error = ""
    #warning = 'null'

    df_imputed["timestamp"] = df_imputed["timestamp"].dt.strftime("%Y-%M-%dT%H:%M:%S%Z+00:00")
    
    json_data = df_imputed.to_dict(orient = "records")

    jsondata = {"comment": msg,"points": json_data,"error":error,"warning":warning}

    return jsondata
