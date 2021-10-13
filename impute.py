from flask import Flask, jsonify, request, make_response, stream_with_context
import json
from flask_restful import Api, Resource, reqparse
import numpy as np
from aqi_usage import input_call
from flask_cors import CORS, cross_origin
import time
import sys
from io import StringIO, BytesIO
import gzip


app = Flask(__name__)
CORS(app, support_credentials=True)

app.config['CORS_HEADERS'] = 'Content-Type'


@app.errorhandler(ValueError)
def server_error(err):
    app.logger.exception(err)
    missing_req_err = "Value Error : Missing Request Data"
    jsondata = {"error":missing_req_err}
    return jsondata 


@app.errorhandler(FileNotFoundError)
def server_error(err):
    app.logger.exception(err)
    file_not_found_err = "Data is not Available for this time series id"
    jsondata = {"error":file_not_found_err}
    return jsondata 


@app.errorhandler(IndexError)
def server_error(err):
    app.logger.exception(err)
    surrogate_not_found = "Error insufficient data. Ensure surrogates are selected."
    jsondata = {"error":surrogate_not_found}
    return jsondata 


@app.route('/v2/predict/discovery-v1',methods=['GET'])
@cross_origin()
def algo_spec():
    name = 'Imputation Algorithm'
    url = 'v2/predict/ImputationAlgorithm'
    description = 'CNN based model'
    minimumRequiredPoints = 4
    
    json_data = {
        "algorithms": [
        {
          "name": name,
          "url": url,
          "description": description,
          
          "settings": 
          {
             "minimumRequiredPoints":minimumRequiredPoints             
          }
        }
                      ]
                }
    return json_data



@app.route('/v2/anomaly/discovery-v1',methods=['GET'])
@cross_origin()
def algo_spec2():
    name = 'ImputationAlgorithm'
    url = '/aqi_impute'
    description = 'CNN based model'
    minimumRequiredPoints = 4
    
    json_data = {
        "algorithms": [
        {
          "name": name,
          "url": url,
          "description": description,
          
          "settings": 
          {
             "minimumRequiredPoints":minimumRequiredPoints
             
          }
        }
                      ]
                }
    return json_data


@app.route('/v2/predict/ImputationAlgorithm',methods=['POST'])
@cross_origin()
def predict():
    data = gzip.decompress(request.data)
    data = eval(data)
#----------------------------------
#   Time Range Values
#----------------------------------
    timeRange = data['timeRange']
    timeRange = list(timeRange.values())
    #timeRange = sorted(timeRange)

#----------------------------------
#   Target Values
#----------------------------------
    target = data['target']
    parameterId = data['target']['parameterId']
    parameterName = data['target']['parameterName']
    locationIdentifier = data['target']['locationIdentifier']
    locationName = data['target']['locationName']
    parameter = data['target']['timeSeriesUniqueId']
    target_station = data['target']['timeSeriesUniqueId']
    
#----------------------------------
#   Surrogate Values
#----------------------------------   
    surrogates = data['surrogates'][0]
    surrogate_locationId = surrogates['locationIdentifier']
    time_uid = data['surrogates'][0]
    surrogate_stations = time_uid['timeSeriesUniqueId']
    surrogate_parameter = data['surrogates'][0]
    surrogate_parameterId = surrogate_parameter['parameterId']
    surrogate_location = data['surrogates'][0]
    surrogate_location_name = surrogate_location['locationName']
    surrogate_parameter = data['surrogates'][0]
    surrogate_parameterName = surrogate_parameter['parameterName']  
#----------------------------------
#   Interval
#----------------------------------  
       
    interval = data['predictSamplePeriodMin']

    predictions = input_call(parameter,parameterName,locationIdentifier,
                        locationName,target_station,timeRange,surrogate_stations,interval,surrogate_locationId,
                        surrogate_location_name,surrogate_parameterName)

    return predictions

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000,debug=True,ssl_context=('cert.pem', 'key.pem'))