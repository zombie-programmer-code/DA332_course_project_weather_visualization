import os
import csv
from datetime import datetime, time, timedelta
from cs50 import SQL
from flask import Flask, render_template, request, jsonify
import pandas as pd
import plotly.express as px
import requests
import numpy as np
import joblib
from datetime import datetime, timedelta
import time as tm  # Alias the time module to avoid conflicts with datetime.time
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
db1 = SQL("sqlite:///live_weather.db")
city_names = ["Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Ahmedabad", 
                      "Chennai", "Kolkata", "Surat", "Pune", "Jaipur", 
                      "Lucknow", "Kanpur", "Nagpur", "Indore", "Thane", 
                      "Bhopal", "Visakhapatnam", "Patna", "Vadodara", "Ghaziabad",
                      "Shimla", "Chandigarh", "Dehradun", "Ranchi", "Raipur", 
                        "Guwahati", "Itanagar", "Kohima", "Aizawl", "Agartala", 
                        "Imphal", "Gangtok", "Bhubaneswar", "Thiruvananthapuram",
                        "Panaji", "Shillong"]

api_key = '126aff7cea9b454ca9c72738253103'
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
                city TEXT NOT NULL,
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

        for city in city_names:
            print(f"Processing data for {city}...")
            try:
                # Read the CSV file for the city
                df = pd.read_csv(f'data/{city}_weather.csv')
                df['Date'] = pd.to_datetime(df['Date'])  # Ensure the 'Date' column is in datetime format

                # Insert data into the weather_data table
                for _, row in df.iterrows():
                    db.execute("""
                        INSERT INTO weather_data (
                            city, date, max_temperature, min_temperature, total_rainfall, 
                            sunrise_time, sunset_time, daylight_duration, 
                            precipitation_hours, max_wind_speed
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, city,  # Add the city name here
                        row['Date'].strftime('%Y-%m-%d'), 
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

api_key_weather = '126aff7cea9b454ca9c72738253103'

def get_historical_hourly_weather(api_key, latitude, longitude, num_hours):
    """
    Fetches historical hourly weather data for a given location and number of hours in the past.
    Implements retry logic if the API rate limit is exceeded.

    Parameters:
    - api_key (str): Your WeatherAPI.com API key.
    - latitude (float): Latitude of the location.
    - longitude (float): Longitude of the location.
    - num_hours (int): Number of hours in the past to retrieve data for.

    Returns:
    - DataFrame: A pandas DataFrame containing the historical weather data.
    """
    if num_hours > 720:  # WeatherAPI.com allows fetching data up to 30 days (720 hours) in the past
        raise ValueError("WeatherAPI.com's History API allows fetching data for up to 30 days (720 hours) in the past.")

    end_time = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    start_time = end_time - timedelta(hours=num_hours)

    base_url = "http://api.weatherapi.com/v1/history.json"
    all_data = []

    current_time = start_time
    while current_time <= end_time:
        date_str = current_time.strftime('%Y-%m-%d')
        params = {
            "key": api_key,
            "q": f"{latitude},{longitude}",
            "dt": date_str,
            "hour": current_time.hour
        }

        while True:
            try:
                response = requests.get(base_url, params=params)
                if response.status_code == 429:
                    print("Rate limit exceeded. Waiting 60 seconds before retrying...")
                    time.sleep(60)
                    continue

                response.raise_for_status()
                data = response.json()

                for hour_data in data.get("forecast", {}).get("forecastday", [])[0].get("hour", []):
                    all_data.append({
                        "Datetime": datetime.strptime(hour_data["time"], '%Y-%m-%d %H:%M'),
                        "Temperature (°C)": hour_data["temp_c"],
                        "Feels Like (°C)": hour_data["feelslike_c"],
                        "Precipitation (mm)": hour_data["precip_mm"],
                        "Wind Speed (kph)": hour_data["wind_kph"],
                        "Wind Direction": hour_data["wind_dir"],
                        "Cloud Cover (%)": hour_data["cloud"],
                        "Humidity (%)": hour_data["humidity"],
                        "Pressure (mb)": hour_data["pressure_mb"]
                    })

                break

            except requests.exceptions.RequestException as e:
                print(f"Request failed: {e}. Retrying in 60 seconds...")
                time.sleep(60)

        current_time += timedelta(hours=1)

    df = pd.DataFrame(all_data)
    df = df[df["Datetime"] <= end_time]
    df = df.tail(num_hours).reset_index(drop=True)

    return df

API_KEY = "67c490b3a3c78893272577xmvd2c31f"
def get_lat_lon(city):
    """
    Fetches latitude and longitude for a given city using Geocode Maps.co API.

    Parameters:
    - city (str): Name of the city
    Returns:
    - tuple: (latitude, longitude) or (None, None) if not found
    """
    url = f"https://geocode.maps.co/search?q={city}&format=json&api_key={API_KEY}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()  
        data = response.json()

        if data:
            lat = float(data[0]["lat"])
            lon = float(data[0]["lon"])
            return lat, lon
        else:
            return None, None

    except requests.exceptions.RequestException as e:
        print(f"⚠ Error fetching data for {city}: {e}")
        return None, None

def get_historical_weather(latitude, longitude, start_date, end_date):
    """
    Fetches historical daily weather data for a given location and date range.
    Implements retry logic if the API rate limit is exceeded.
    
    Additional daily parameters include:
      - daylight_duration (converted to HH:MM:SS)
      - precipitation_hours
      - windspeed_10m_max
      - uv_index_max
    """
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        # Added additional parameters to the daily endpoint
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,sunrise,sunset,daylight_duration,precipitation_hours,windspeed_10m_max,uv_index_max",
        "timezone": "auto"
    }
    i = 0
    while True:
        try:
            response = requests.get(base_url, params=params)
            if response.status_code == 429:
                print("Rate limit exceeded. Waiting 60 seconds before retrying...")
                time.sleep(60)  # Wait before retrying
                continue  # Retry the request
            
            response.raise_for_status()  # Raise exceptions for other errors
            data = response.json()

            # Format sunrise and sunset times to HH:MM:SS format
            sunrise_times = [pd.to_datetime(t).strftime('%H:%M:%S') for t in data["daily"]["sunrise"]]
            sunset_times = [pd.to_datetime(t).strftime('%H:%M:%S') for t in data["daily"]["sunset"]]
            # Convert daylight duration from seconds to HH:MM:SS format
            # Convert daylight duration from seconds to HH:MM:SS format
            daylight_durations = [tm.strftime('%H:%M:%S', tm.gmtime(int(d))) for d in data["daily"]["daylight_duration"]]            
            daily_df = pd.DataFrame({
                "Date": pd.to_datetime(data["daily"]["time"]),
                "Max Temperature (°C)": data["daily"]["temperature_2m_max"],
                "Min Temperature (°C)": data["daily"]["temperature_2m_min"],
                "Total Rainfall (mm)": data["daily"]["precipitation_sum"],
                "Sunrise Time": sunrise_times,
                "Sunset Time": sunset_times,
                "Daylight Duration": daylight_durations,
                "Precipitation Hours": data["daily"]["precipitation_hours"],
                "Max Wind Speed (m/s)": data["daily"]["windspeed_10m_max"],
                "Max UV Index": data["daily"]["uv_index_max"]
            })

            return daily_df  # Successfully fetched data, return DataFrame

        except requests.exceptions.RequestException as e:
            i = i + 1
            print(f"Request failed: {e}. Retrying in 60 seconds...")
            tm.sleep(60)  # Wait and retry
            if i >= 2:
                print("Failed to fetch data after multiple attempts.")
                return None



# Define the route for the homepage
@app.route('/')
def index():
    return render_template('home.html')

@app.route('/historical_trends', methods=['GET', 'POST'])
def historical_trends():
    if request.method == 'POST':
        # Line plot inputs
        line_cities = request.form.getlist('line_cities')
        line_from_year = int(request.form['line_from_year'])
        line_to_year = int(request.form['line_to_year'])
        line_variables = request.form.getlist('line_variables')

        # Box plot inputs
        box_cities = request.form.getlist('box_cities')
        box_variable = request.form['box_variable']

        # Handle grouped "temperature" option for line plot
        if "temperature" in line_variables:
            line_variables.remove("temperature")
            line_variables.extend(["max_temperature", "min_temperature"])

        # Query the database for line plot
        line_query = f"""
            SELECT city, date, {', '.join(line_variables)}
            FROM weather_data
            WHERE city IN ({', '.join(['?'] * len(line_cities))})
            AND strftime('%Y', date) BETWEEN ? AND ?
        """
        line_rows = db.execute(line_query, *line_cities, str(line_from_year), str(line_to_year))
        line_df = pd.DataFrame(line_rows)

        # Query the database for box plot
        box_query = f"""
            SELECT city, date, {box_variable}
            FROM weather_data
            WHERE city IN ({', '.join(['?'] * len(box_cities))})
        """
        box_rows = db.execute(box_query, *box_cities)
        box_df = pd.DataFrame(box_rows)

        # Check for empty data
        if line_df.empty and box_df.empty:
            return render_template('historical_trends.html', city_names=city_names, error="No data found for the selected cities and year range.")

        # Generate the line plot
        line_fig = px.line(
            line_df,
            x='date',
            y=line_variables,
            color='city',
            labels={'value': 'Weather Metrics', 'variable': 'Metric'},
            title=f"Weather Trends (Line Plot) for {', '.join(line_cities)} ({line_from_year} - {line_to_year})"
        )

        # Generate the box plot
        box_fig = px.box(
            box_df,
            x='city',
            y=box_variable,
            color='city',
            labels={'value': 'Weather Metrics', 'variable': 'Metric'},
            title=f"Box Plot for {box_variable} in {', '.join(box_cities)}"
        )

        # Convert the plots to HTML
        line_plot_html = line_fig.to_html(full_html=False)
        box_plot_html = box_fig.to_html(full_html=False)

        # Render the plots in the HTML template
        return render_template(
            'historical_trends_plot.html',
            line_plot_html=line_plot_html,
            box_plot_html=box_plot_html
        )

    # Render the form if the request method is GET
    return render_template('historical_trends.html', city_names=city_names)

@app.route('/forecasts')
def forecasts():
    # Logic for forecasts
    return "Forecasts Page (to be implemented)"



@app.route('/live_weather', methods=['GET', 'POST'])
def live_weather():
    if request.method == 'POST':
        city = request.form['city'].strip()
        hours = int(request.form['hours'])

        # Get lat/lon
        lat, lon = get_lat_lon(city)
        if lat is None or lon is None:
            return render_template('live_weather.html', error=f"Could not fetch coordinates for {city}.")

        # Create the hourly_weather table if it doesn't exist
        db1.execute("""
            CREATE TABLE IF NOT EXISTS hourly_weather (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                city TEXT NOT NULL,
                datetime TEXT NOT NULL,
                temperature REAL,
                feels_like REAL,
                precipitation REAL,
                wind_speed REAL,
                wind_direction TEXT,
                cloud_cover REAL,
                humidity REAL,
                pressure REAL
            )
        """)

        # Time range
        end_time = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        start_time = end_time - timedelta(hours=hours)

        # Check existing data
        rows = db1.execute(
            "SELECT * FROM hourly_weather WHERE city = ? AND datetime BETWEEN ? AND ? ORDER BY datetime",
            city, start_time.strftime('%Y-%m-%d %H:%M'), end_time.strftime('%Y-%m-%d %H:%M')
        )
        df = pd.DataFrame(rows)

        # Identify missing timestamps
        existing_times = set(pd.to_datetime(df['datetime'])) if not df.empty else set()
        expected_times = set(start_time + timedelta(hours=i) for i in range(hours))
        missing_times = sorted(expected_times - existing_times)

        # Fetch and insert missing data
        if missing_times:
            print(f"Missing {len(missing_times)} hours for {city}. Fetching from API...")

            api_df = get_historical_hourly_weather(api_key_weather, lat, lon, hours)
            if api_df is None or api_df.empty:
                return render_template('live_weather.html', error="Could not fetch hourly weather data.")

            api_df = api_df[api_df["Datetime"].isin(missing_times)]

            for _, row in api_df.iterrows():
                db1.execute("""
                    INSERT INTO hourly_weather (
                        city, datetime, temperature, feels_like, precipitation, wind_speed, 
                        wind_direction, cloud_cover, humidity, pressure
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, city, row['Datetime'].strftime('%Y-%m-%d %H:%M'), row['Temperature (°C)'], row['Feels Like (°C)'],
                    row['Precipitation (mm)'], row['Wind Speed (kph)'], row['Wind Direction'],
                    row['Cloud Cover (%)'], row['Humidity (%)'], row['Pressure (mb)'])

            # Reload after insert
            rows = db1.execute(
                "SELECT * FROM hourly_weather WHERE city = ? AND datetime BETWEEN ? AND ? ORDER BY datetime",
                city, start_time.strftime('%Y-%m-%d %H:%M'), end_time.strftime('%Y-%m-%d %H:%M')
            )
            df = pd.DataFrame(rows)

        if df.empty:
            return render_template('live_weather.html', error="No hourly data found.")

        # Render Plotly table
        import plotly.graph_objects as go
        df['datetime'] = pd.to_datetime(df['datetime'])

        fig = go.Figure(data=[go.Table(
            header=dict(
                values=["Datetime", "Temperature (°C)", "Feels Like (°C)", "Precipitation (mm)",
                        "Wind Speed (kph)", "Wind Direction", "Cloud Cover (%)",
                        "Humidity (%)", "Pressure (mb)"],
                fill_color='paleturquoise',
                align='left'
            ),
            cells=dict(
                values=[
                    df['datetime'].dt.strftime('%Y-%m-%d %H:%M'),
                    df['temperature'],
                    df['feels_like'],
                    df['precipitation'],
                    df['wind_speed'],
                    df['wind_direction'],
                    df['cloud_cover'],
                    df['humidity'],
                    df['pressure']
                ],
                fill_color='lavender',
                align='left'
            )
        )])

        table_html = fig.to_html(full_html=False)
        return render_template('live_weather_plot.html', plot_html=table_html, city=city, hours=hours)

    # Render the input form
    return render_template('live_weather.html')


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)