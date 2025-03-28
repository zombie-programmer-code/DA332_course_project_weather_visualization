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
db = SQL("sqlite:///weather.db")

@app.before_request
def store_csv_in_database():
    if not hasattr(app, 'db_initialized'):
        app.db_initialized = True  # Set a flag to ensure this runs only once

        # Check if the weather_data table exists and is not empty
        table_exists = db.execute("""
            SELECT name FROM sqlite_master WHERE type='table' AND name='weather_data'
        """)
        if table_exists:
            row_count = db.execute("SELECT COUNT(*) AS count FROM weather_data")[0]['count']
            if row_count > 0:
                print("Database already contains data. Skipping data insertion.")
                return

        # Create the weather_data table if it doesn't exist
        db.execute("""
            CREATE TABLE IF NOT EXISTS weather_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                max_temperature REAL,
                min_temperature REAL,
                total_rainfall REAL,
                sunrise_time TEXT,
                sunset_time TEXT,
                daylight_duration TEXT,
                precipitation_hours REAL,
                max_wind_speed REAL
            )
        """)

        city_names = ["Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Ahmedabad", 
                      "Chennai", "Kolkata", "Surat", "Pune", "Jaipur", 
                      "Lucknow", "Kanpur", "Nagpur", "Indore", "Thane", 
                      "Bhopal", "Visakhapatnam", "Patna", "Vadodara", "Ghaziabad",
                      "Shimla", "Chandigarh", "Dehradun"]

        for city in city_names:
            try:
                # Read the CSV file for the city
                df = pd.read_csv(f'data/{city}_weather.csv')
                df['Date'] = pd.to_datetime(df['Date'])  # Ensure the 'Date' column is in datetime format

                # Insert data into the weather_data table
                for _, row in df.iterrows():
                    db.execute("""
                        INSERT INTO weather_data (
                            date, max_temperature, min_temperature, total_rainfall, 
                            sunrise_time, sunset_time, daylight_duration, 
                            precipitation_hours, max_wind_speed
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, row['Date'].strftime('%Y-%m-%d'), 
                        row.get('Max Temperature (°C)', None), 
                        row.get('Min Temperature (°C)', None), 
                        row.get('Total Rainfall (mm)', None), 
                        row.get('Sunrise Time', None), 
                        row.get('Sunset Time', None), 
                        row.get('Daylight Duration', None), 
                        row.get('Precipitation Hours', None), 
                        row.get('Max Wind Speed (m/s)', None))
            except FileNotFoundError:
                print(f"data/{city}_weather.csv not found. Skipping...")
            except Exception as e:
                print(f"Error processing {city}: {e}")

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
