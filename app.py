import os
import csv
from datetime import datetime, time, timedelta
from cs50 import SQL
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import joblib
import numpy as np
import pandas as pd
from datetime import timedelta
#from tensorflow.python.keras.models import load_model
#import tensorflow as tf
app = Flask(__name__)
@app.before_request
def initialize():
    
    try:
        df = pd.read_csv(f'data/DA332_project_data/weather_csv/Guwahati_weather.csv')
        df['date'] = pd.to_datetime(df['date'])  # Ensure the 'date' column is in datetime format
    except FileNotFoundError:
        raise FileNotFoundError(f"Guwahati_weather.csv not found in {DATA_PATH}. Please provide the file.")

# Define the route for the homepage
@app.route('/')
def index():
    return render_template('home.html')

@app.route('/historical_trends')
def historical_trends():
    # Logic for historical trends
    return "Historical Trends Page (to be implemented)"

@app.route('/forecasts')
def forecasts():
    # Logic for forecasts
    return "Forecasts Page (to be implemented)"

@app.route('/live_weather')
def live_weather():
    # Logic for live weather conditions
    return "Live Weather Conditions Page (to be implemented)"
