FROM python:3.7

WORKDIR /app
COPY . /app

VOLUME /home/ubuntu/Data/AQI/corrected_df

COPY aqi_algo_api.py ./aqi_algo_api.py

COPY aqi_data.py ./aqi_data.py

COPY aqi_impute_gap.py ./aqi_impute_gap.py

COPY aqi_impute_levels.py ./aqi_impute_levels.py

COPY aqi_impute_series.py ./aqi_impute_series.py

COPY aqi_model.py ./aqi_model.py

COPY aqi_train.py ./aqi_train.py

COPY aqi_usage.py ./aqi_usage.py

COPY aqi_utils.py ./aqi_utils.py		
RUN pip install -r requirements.txt
EXPOSE 5000

CMD gunicorn --certfile cert.pem --keyfile key.pem --workers 3 --bind 0.0.0.0:5000 -m 007 wsgi:app
