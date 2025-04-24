import os
import csv
from datetime import datetime, time, timedelta
from cs50 import SQL
from flask import Flask, render_template, request, jsonify, redirect,session
import pandas as pd
import plotly.express as px
import requests
import numpy as np
import joblib
from math import radians, sin, cos, sqrt, atan2

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
from tensorflow.python.keras.models import load_model
import tensorflow as tf
from timezonefinder import TimezoneFinder
import pytz
app = Flask(__name__)
import secrets
app.secret_key = secrets.token_hex(16)
db = SQL("sqlite:///weather.db")
db1 = SQL("sqlite:///live_weather_map.db")
db2 = SQL("sqlite:///city_lat_long.db")
city_names = ["Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Ahmedabad", 
                      "Chennai", "Kolkata", "Surat", "Pune", "Jaipur", 
                      "Lucknow", "Kanpur", "Nagpur", "Indore", "Thane", 
                      "Bhopal", "Visakhapatnam", "Patna", "Vadodara", "Ghaziabad",
                      "Shimla", "Chandigarh", "Dehradun", "Ranchi", "Raipur", 
                        "Guwahati", "Itanagar", "Kohima", "Aizawl", "Agartala", 
                        "Imphal", "Gangtok", "Bhubaneswar", "Thiruvananthapuram",
                        "Panaji", "Shillong"]

api_key = '126aff7cea9b454ca9c72738253103'
world_cities = pd.read_csv('data/world_cities_lat_long.csv')

# Extract unique country names
countries = world_cities['country'].unique().tolist()
def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on the Earth using the Haversine formula.
    """
    R = 6371  # Radius of the Earth in kilometers
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

@app.before_request
def store_csv_in_database():
    #populate_lat_long_table()
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
def convert_utc_to_local(dt_utc, lat, lon):
    """
    Converts a UTC datetime to the local timezone based on latitude and longitude.
    """
    tf = TimezoneFinder()
    timezone_str = tf.timezone_at(lat=lat, lng=lon)
    
    if timezone_str is None:
        raise ValueError("Could not determine timezone from lat/lon.")

    local_tz = pytz.timezone(timezone_str)
    
    # Check if the datetime is naive or already timezone-aware
    if dt_utc.tzinfo is None:
        # Localize the naive datetime to UTC
        dt_utc = pytz.utc.localize(dt_utc)
    
    # Convert to local timezone
    dt_local = dt_utc.astimezone(local_tz)
    return dt_local
import requests
import pandas as pd
import time

import requests
import pandas as pd
import time
from datetime import datetime, timedelta

def get_historical_weather(latitude, longitude):
    """
    Fetches 7 unique, non-NaN days of historical weather data going backward from today.
    Skips any days that contain missing data (NaNs).
    """
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    collected_data = []
    collected_dates = set()
    current_day = datetime.today()

    while len(collected_dates) < 7:
        start_date = (current_day - timedelta(days=2)).strftime('%Y-%m-%d')
        end_date = (current_day - timedelta(days=1)).strftime('%Y-%m-%d')

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max",
            "timezone": "auto"
        }

        try:
            response = requests.get(base_url, params=params)
            if response.status_code == 429:
                print("Rate limit exceeded. Waiting 60 seconds before retrying...")
                time.sleep(60)
                continue

            response.raise_for_status()
            data = response.json()

            if "daily" in data and "time" in data["daily"]:
                for i, date_str in enumerate(data["daily"]["time"]):
                    row = {
                        "Date": pd.to_datetime(date_str),
                        "Max Temperature (°C)": data["daily"]["temperature_2m_max"][i],
                        "Min Temperature (°C)": data["daily"]["temperature_2m_min"][i],
                        "Total Rainfall (mm)": data["daily"]["precipitation_sum"][i],
                        "Max Wind Speed (m/s)": data["daily"]["windspeed_10m_max"][i]
                    }

                    # Skip if any NaNs are present
                    if pd.isna(list(row.values())).any():
                        continue

                    if row["Date"] not in collected_dates:
                        collected_data.append(row)
                        collected_dates.add(row["Date"])

            current_day -= timedelta(days=1)

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}. Retrying in 60 seconds...")
            time.sleep(60)

    full_df = pd.DataFrame(collected_data).sort_values(by="Date").reset_index(drop=True)
    return full_df

def fetch_weather_for_date(latitude, longitude, date):
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": date,
        "end_date": date,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max",
        "timezone": "auto"
    }

    while True:
        try:
            response = requests.get(base_url, params=params)
            if response.status_code == 429:
                print("Rate limit hit. Waiting 60 seconds...")
                time.sleep(60)
                continue
            response.raise_for_status()
            data = response.json()

            if "daily" in data and "time" in data["daily"]:
                return pd.DataFrame({
                    "Date": pd.to_datetime(data["daily"]["time"]),
                    "Max Temperature (°C)": data["daily"]["temperature_2m_max"],
                    "Min Temperature (°C)": data["daily"]["temperature_2m_min"],
                    "Total Rainfall (mm)": data["daily"]["precipitation_sum"],
                    "Max Wind Speed (m/s)": data["daily"]["windspeed_10m_max"]
                })
            else:
                return None
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}. Retrying...")
            time.sleep(60)


def rainfall_category_to_mm(category):
    if category == 0:
        return 0.0
    elif category == 1:
        return round(np.random.exponential(scale=1.5), 2) % 5
    elif category == 2:
        value = 5 + np.random.exponential(scale=3)
        return round(min(value, 19.99), 2)
    elif category == 3:
        value = 20 + np.random.exponential(scale=15)
        return round(min(value, 199.99), 2)


def create_single_lagged_tuple(df):
    if len(df) != 7:
        raise ValueError("Expected exactly 7 rows of input")

    last_day = df.iloc[-1]
    day_of_week = last_day['Date'].dayofweek
    sin_day = np.sin(2 * np.pi * day_of_week / 7)
    cos_day = np.cos(2 * np.pi * day_of_week / 7)

    lagged_block = df[[
        "Max Temperature (°C)", 
        "Min Temperature (°C)", 
        "Total Rainfall (mm)", 
        "Max Wind Speed (m/s)"
    ]].values.flatten()

    return np.concatenate([lagged_block, [sin_day, cos_day]])


from tensorflow.keras.models import load_model  # type: ignore
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def rolling_weather_prediction(latitude, longitude, model_path, scaler_path, n_days):
    model = load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)

    df = get_historical_weather(latitude, longitude)
    print("Initial data:", df)

    history = []

    for step in range(n_days):
        # Prepare input features
        X_raw = create_single_lagged_tuple(df).reshape(1, -1)
        X_scaled = scaler.transform(X_raw)

        # Predict
        reg_output, class_output = model.predict(X_scaled, verbose=0)
        max_temp, min_temp, wind_speed = reg_output[0]

        # Rainfall prediction
        rainfall_class = np.argmax(class_output[0])
        rainfall_mm = rainfall_category_to_mm(rainfall_class)

        # Prediction date = today + step
        next_date = datetime.today().date() + timedelta(days=step)

        # Store result
        history.append({
            "Prediction Date": next_date,
            "Predicted Max Temp (°C)": round(max_temp, 2),
            "Predicted Min Temp (°C)": round(min_temp, 2),
            "Predicted Rainfall (mm)": round(rainfall_mm, 2),
            "Predicted Wind Speed (m/s)": round(wind_speed, 2),
            "Rainfall Category": ["No Rain", "Light Rain", "Moderate Rain", "Heavy Rain"][rainfall_class]
        })

        # Simulate next day in place of real API data
        simulated_row = {
            "Date": pd.to_datetime(next_date),
            "Max Temperature (°C)": max_temp,
            "Min Temperature (°C)": min_temp,
            "Total Rainfall (mm)": rainfall_mm,
            "Max Wind Speed (m/s)": wind_speed
        }

        df = pd.concat([df.iloc[1:], pd.DataFrame([simulated_row])], ignore_index=True)

    return pd.DataFrame(history)


def get_historical_hourly_weather(api_key, latitude, longitude, end_time, start_time, num_hours):
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

    base_url = "http://api.weatherapi.com/v1/history.json"
    all_data = []

    # Convert UTC time range to local time range
    local_start_time = convert_utc_to_local(start_time, latitude, longitude)
    local_end_time = convert_utc_to_local(end_time, latitude, longitude)

    print(f"Converted UTC time range to local time range: {local_start_time} - {local_end_time}")

    current_time = local_start_time
    while current_time <= local_end_time:
        date_str = current_time.strftime('%Y-%m-%d')
        params = {
            "key": api_key,
            "q": f"{latitude},{longitude}",
            "dt": date_str,
            "hour": current_time.hour  # Use the local hour
        }

        while True:
            try:
                response = requests.get(base_url, params=params)
                if response.status_code == 429:
                    print("Rate limit exceeded. Waiting 60 seconds before retrying...")
                    tm.sleep(60)
                    continue

                response.raise_for_status()
                data = response.json()

                # Check if forecast data is available
                forecast_days = data.get("forecast", {}).get("forecastday", [])
                if not forecast_days:
                    print(f"No forecast data available for {date_str} at location {latitude}, {longitude}. Skipping...")
                    break  # Skip to the next iteration if no forecast data is available

                for hour_data in forecast_days[0].get("hour", []):
                    # Parse the local time from the API
                    local_time = datetime.strptime(hour_data["time"], '%Y-%m-%d %H:%M')

                    # Convert local time to UTC for consistency
                    local_time_zone = pytz.timezone(hour_data.get("tz_id", "UTC"))  # Use the time zone ID from the API if available
                    local_time = local_time_zone.localize(local_time)  # Localize the naive datetime
                    utc_time = local_time.astimezone(pytz.utc)  # Convert to UTC

                    this_hour = {
                        "Datetime": utc_time,
                        "Temperature (°C)": hour_data["temp_c"],
                        "Feels Like (°C)": hour_data["feelslike_c"],
                        "Precipitation (mm)": hour_data["precip_mm"],
                        "Wind Speed (kph)": hour_data["wind_kph"],
                        "Wind Direction": hour_data["wind_dir"],
                        "Cloud Cover (%)": hour_data["cloud"],
                        "Humidity (%)": hour_data["humidity"],
                        "Pressure (mb)": hour_data["pressure_mb"]
                    }
                    all_data.append(this_hour)
                    print(this_hour)
                break

            except requests.exceptions.RequestException as e:
                print(f"Request failed: {e}. Retrying in 60 seconds...")
                tm.sleep(60)

        current_time += timedelta(hours=1)

    if not all_data:
        print(f"No data collected for location {latitude}, {longitude}. Returning an empty DataFrame.")
        return pd.DataFrame()  # Return an empty DataFrame if no data was collected

    df = pd.DataFrame(all_data)
    # Keep only the last `num_hours` rows
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

def populate_lat_long_table():
    if not hasattr(app, 'db_initialized'):
        app.db_initialized = True  # Set a flag to ensure this runs only once

        # Check if the weather_data table exists and is not empty
        table_exists = db2.execute("""
            SELECT name FROM sqlite_master WHERE type='table' AND name='city_lat_long'
        """)
        if table_exists:
            row_count = db2.execute("SELECT COUNT(*) AS count FROM city_lat_long")[0]['count']
            if row_count > 0:
                print("Database already contains data. Skipping data insertion.")
                return

        # Create the weather_data table if it doesn't exist
        db2.execute("""
        CREATE TABLE IF NOT EXISTS city_lat_long (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            city TEXT NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL
            )
        """)

        for city in city_names:
            print(f"Processing data for {city}...")
            try:
                # Get lat/lon
                lat, lon = get_lat_lon(city)
                if lat is None or lon is None:
                    print(f"Could not fetch coordinates for {city}.")
                    continue

                # Insert data into the city_lat_long table
                db2.execute("""
                    INSERT INTO city_lat_long (city, latitude, longitude) VALUES (?, ?, ?)
                """, city, lat, lon)
            except Exception as e:
                print(f"Error processing {city}: {e}")


# Define the route for the homepage
@app.route('/')
def index():
    return render_template('home.html')


@app.route('/check_status')
def check_status():
    page = request.args.get('page', '')
    # Check if the page is ready in the session
    is_ready = session.get(f'{page}_ready', True)
    return jsonify({'ready': is_ready})
    
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

@app.route('/view_pre_generated_statistics', methods=['GET'])
def view_pre_generated_statistics():
    # If the loading flag is not complete, render the suspense page.
    if request.args.get('loading') != 'complete':
        return render_template('suspense_fast.html', destination='view_pre_generated_statistics')
    
    # Otherwise (loading is complete), continue to build the plots.
    region_month_stats = pd.read_csv('data/region_month_stats.csv')
    print("work in progress")
    
    # Map numeric months to month names
    month_mapping = {
        1: "January", 2: "February", 3: "March", 4: "April",
        5: "May", 6: "June", 7: "July", 8: "August",
        9: "September", 10: "October", 11: "November", 12: "December"
    }
    region_month_stats['Month'] = region_month_stats['Month'].map(month_mapping)

    # Generate the maximum temperature plot
    max_plot_fig = px.bar(
        region_month_stats,
        x='Region',
        y='Max Temperature (°C)_mean',
        error_y='Max Temperature (°C)_std',
        color='Region',
        facet_col='Month',
        facet_col_wrap=4,
        title="Monthly Maximum Temperature with Standard Deviations Highlighted"
    )
    max_plot_fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    max_plot_fig.update_yaxes(title_text=None)
    max_plot_fig.update_layout(height=600)
    max_plot_html = max_plot_fig.to_html(full_html=False)

    # Generate the minimum temperature plot
    min_plot_fig = px.bar(
        region_month_stats,
        x='Region',
        y='Min Temperature (°C)_mean',
        error_y='Min Temperature (°C)_std',
        color='Region',
        facet_col='Month',
        facet_col_wrap=4,
        title="Monthly Minimum Temperature with Standard Deviations Highlighted"
    )
    min_plot_fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    min_plot_fig.update_yaxes(title_text=None)
    min_plot_fig.update_layout(height=600)
    min_plot_html = min_plot_fig.to_html(full_html=False)

    # Generate the rainfall plot
    rain_fig = px.bar(
        region_month_stats,
        x='Region',
        y='Total Rainfall (mm)_mean',
        error_y='Total Rainfall (mm)_std',
        color='Region',
        facet_col='Month',
        facet_col_wrap=4,
        title="Monthly Rainfall (mm) with Standard Deviations Highlighted"
    )
    rain_fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    rain_fig.update_yaxes(title_text=None)
    rain_fig.update_layout(height=600)
    rain_plot_html = rain_fig.to_html(full_html=False)

    # Render the final template with the plots.
    return render_template(
        'historical_statistics.html',
        max_plot_html=max_plot_html,
        min_plot_html=min_plot_html,
        rain_plot_html=rain_plot_html
    )

@app.route('/forecasts')
def forecasts():
    # Logic for forecasts
    return "Forecasts Page (to be implemented)"

@app.route('/Temperature_today')
def today():
    # Logic for temperature today
    if request.args.get('loading') != 'complete':
        return render_template('suspense_fast.html', destination='Temperature_today')
    return render_template('Today.html')

@app.route('/Live-Weather-Map')
def map():
    # Logic for temperature today
    if request.args.get('loading') != 'complete':
        return render_template('suspense_fast.html', destination='Live-Weather-Map')
    return render_template('map.html')

@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    # Load the world_cities_lat_long file
    world_cities = pd.read_csv('data/world_cities_lat_long.csv')

    # Get the query parameter
    query = request.args.get('q', '').lower()

    # Filter city names that match the query
    suggestions = world_cities[world_cities['city'].str.lower().str.startswith(query)]['city'].unique()

    # Return the suggestions as a JSON response
    return jsonify(list(suggestions))

def get_sunrise_sunset_times(lat, lon):
    """
    Fetches sunrise and sunset times for a given latitude and longitude using the Sunrise-Sunset API.
    """
    url = "https://api.sunrise-sunset.org/json"
    params = {
        "lat": lat,
        "lng": lon,
        "formatted": 0  # Use ISO 8601 format for easier parsing
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    sunrise = datetime.fromisoformat(data['results']['sunrise'])
    sunset = datetime.fromisoformat(data['results']['sunset'])

    return sunrise, sunset


@app.route('/live_weather_map', methods=['GET'])
def live_weather_map():
    # Create the weather_map_data table if it doesn't exist
    db1.execute("""
        CREATE TABLE IF NOT EXISTS weather_map_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            city TEXT NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            temperature REAL,
            feels_like REAL,
            humidity REAL,
            precipitation REAL,
            cloud_cover REAL,
            utc_time TEXT NOT NULL,
            sunrise TEXT,
            sunset TEXT
        )
    """)
    flag = 0
    # Load the world_cities_lat_long file
    selected_cities = pd.read_csv('data/world_cities_map.csv')

    # Prepare data for the map
    map_data = []
    current_utc_time = datetime.utcnow().replace(minute=0, second=0, microsecond=0)

    for _, row in selected_cities.iterrows():
        city = row['city']
        lat = row['latitude']
        lon = row['longitude']

        city = str(city)

        current_utc_time_str = current_utc_time.strftime('%Y-%m-%d %H:%M:%S')

        # Use the formatted string in the query
        existing_data = db1.execute(
            "SELECT * FROM weather_map_data WHERE city = ? AND utc_time = ?",
            city, current_utc_time_str
        )

        if existing_data:
            # Convert UTC time to local time
            local_time = convert_utc_to_local(current_utc_time, lat, lon)

            # Fetch sunrise and sunset times
            #sunrise, sunset = get_sunrise_sunset_times(lat, lon)
            sunrise = existing_data[0]['sunrise']
            sunset = existing_data[0]['sunset']
            # Convert sunrise and sunset to datetime objects
            sunrise = datetime.strptime(sunrise, '%Y-%m-%d %H:%M:%S')
            sunset = datetime.strptime(sunset, '%Y-%m-%d %H:%M:%S')

            # Assign the same timezone as local_time
            sunrise = sunrise.replace(tzinfo=local_time.tzinfo)
            sunset = sunset.replace(tzinfo=local_time.tzinfo)
            # Determine if the sun is up
            weather_symbol = "☀️" if existing_data[0]['cloud_cover'] < 40 else (
                "🌧️" if existing_data[0]['precipitation'] > 0 else "☁️"
            )
            if not (sunrise <= local_time <= sunset):  # If it's not daytime
                if weather_symbol == "☀️":
                    weather_symbol = "🌙"

            map_data.append({
                'city': city,
                'lat': lat,
                'lon': lon,
                'temperature': existing_data[0]['temperature'],
                'feels_like': existing_data[0]['feels_like'],
                'humidity': existing_data[0]['humidity'],
                'precipitation': existing_data[0]['precipitation'],
                'cloud_cover': existing_data[0]['cloud_cover'],
                'local_time': local_time.strftime('%Y-%m-%d %H:%M:%S'),
                'sunrise': existing_data[0]['sunrise'],
                'sunset': existing_data[0]['sunset'],
                'weather_symbol': weather_symbol
            })
            flag = 1
        else:
            weather_data = get_historical_hourly_weather(
                api_key_weather, lat, lon, current_utc_time, current_utc_time - timedelta(hours=1), 1
            )
            if not weather_data.empty:
                # Extract the first row of weather data
                weather_row = weather_data.iloc[0]
                temperature = weather_row['Temperature (°C)']
                feels_like = weather_row['Feels Like (°C)']
                humidity = float(weather_row['Humidity (%)'])
                precipitation = weather_row['Precipitation (mm)']
                cloud_cover = float(weather_row['Cloud Cover (%)'])

                # Convert UTC time to local time
                local_time = convert_utc_to_local(current_utc_time, lat, lon)

                # Fetch sunrise and sunset times
                sunrise, sunset = get_sunrise_sunset_times(lat, lon)

                # Convert sunrise and sunset to local time
                sunrise = convert_utc_to_local(sunrise, lat, lon)
                sunset = convert_utc_to_local(sunset, lat, lon)     
                weather_symbol = "☀️" if cloud_cover < 40 else (
                    "🌧️" if precipitation > 0 else "☁️"
                )
                if not (sunrise <= local_time <= sunset):  # If it's not daytime
                    if weather_symbol == "☀️":
                        weather_symbol = "🌙"

                print(f"Inserting data for city: {city}")
                print(f"Data types: city={type(city)}, lat={type(lat)}, lon={type(lon)}, "
                      f"temperature={type(temperature)}, feels_like={type(feels_like)}, "
                      f"humidity={type(humidity)}, precipitation={type(precipitation)}, "
                      f"cloud_cover={type(cloud_cover)}, utc_time={type(current_utc_time.strftime('%Y-%m-%d %H:%M:%S'))}")

                # Store the data in the database
                db1.execute("""
                    INSERT INTO weather_map_data (
                        city, latitude, longitude, temperature, humidity, precipitation, utc_time, cloud_cover, feels_like, sunrise, sunset
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, city, lat, lon, temperature, humidity, precipitation, current_utc_time.strftime('%Y-%m-%d %H:%M:%S'), cloud_cover, feels_like, sunrise.strftime('%Y-%m-%d %H:%M:%S'), sunset.strftime('%Y-%m-%d %H:%M:%S'))

                # Add the data to the map
                map_data.append({
                    'city': city,
                    'lat': lat,
                    'lon': lon,
                    'temperature': temperature,
                    'feels_like': feels_like,
                    'humidity': humidity,
                    'precipitation': precipitation,
                    'cloud_cover': cloud_cover,
                    'local_time': local_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'sunrise': sunrise.strftime('%Y-%m-%d %H:%M:%S'),
                    'sunset': sunset.strftime('%Y-%m-%d %H:%M:%S'),
                    'weather_symbol': weather_symbol
                })

    # Convert to DataFrame
    map_df = pd.DataFrame(map_data)
    print(map_df)
    # Generate the map
    if flag == 1:
        import plotly.express as px
        fig = px.scatter_mapbox(
            map_df,
            lat='lat',
            lon='lon',
            hover_name='city',
            hover_data={
                'temperature': True,
                'feels_like': True,
                'humidity': True,
                'precipitation': True,
                'cloud_cover': True,
                'local_time': True,
                'sunrise': True,
                'sunset': True,
                'weather_symbol': True
            },
            color='temperature',
            size='humidity',
            color_continuous_scale='Viridis',
            title='Weather Data for 100 Cities (Last Hour)'
        )
        fig.update_layout(mapbox_style='carto-positron', mapbox_zoom=2, mapbox_center={'lat': 20, 'lon': 0})
        fig.update_layout(margin={'r': 0, 't': 0, 'l': 0, 'b': 0})

        current_utc_time = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        # Convert the map to HTML
        map_html = fig.to_html(full_html=False)
        return render_template('live_weather_map.html', map_html=map_html, last_updated=current_utc_time.strftime('%Y-%m-%d %H:%M:%S UTC'))
    else:
        # Redirect back to the live_weather_map route
        return redirect('/live_weather_map')

@app.route('/autocomplete_countries', methods=['GET'])
def autocomplete_countries():
    # Get the query parameter
    query = request.args.get('q', '').strip().lower()

    # Filter country names that match the query
    suggestions = [country for country in countries if country.lower().startswith(query)]

    # Return the suggestions as a JSON response
    return jsonify(suggestions)

@app.route('/country_weather', methods=['GET', 'POST'])
def country_weather():
    if request.method == 'POST':
        country = request.form['country'].strip()

        # Load the world_cities_lat_long.csv file
        world_cities = pd.read_csv('data/world_cities_lat_long.csv')

        # Filter cities for the given country
        country_cities = world_cities[world_cities['country'].str.lower() == country.lower()]

        if country_cities.empty:
            return render_template('country_weather.html', error=f"No cities found for the country: {country}")

        # Limit to a maximum of 10 cities
        country_cities = country_cities.head(10)

        # Prepare data for the map
        map_data = []
        current_utc_time = datetime.utcnow().replace(minute=0, second=0, microsecond=0)

        for _, row in country_cities.iterrows():
            city = row['city']
            lat = row['latitude']
            lon = row['longitude']

            # Check if data is already in the database
            existing_data = db1.execute(
                "SELECT * FROM weather_map_data WHERE city = ? AND utc_time = ?",
                city, current_utc_time.strftime('%Y-%m-%d %H:%M:%S')
            )

            if existing_data:
                # Convert UTC time to local time
                local_time = convert_utc_to_local(current_utc_time, lat, lon)

                # Fetch sunrise and sunset times
                sunrise = existing_data[0]['sunrise']
                sunset = existing_data[0]['sunset']
                # Convert sunrise and sunset to datetime objects
                sunrise = datetime.strptime(sunrise, '%Y-%m-%d %H:%M:%S')
                sunset = datetime.strptime(sunset, '%Y-%m-%d %H:%M:%S')

                # Assign the same timezone as local_time
                sunrise = sunrise.replace(tzinfo=local_time.tzinfo)
                sunset = sunset.replace(tzinfo=local_time.tzinfo)
                
                # Determine if the sun is up
                weather_symbol = "☀️" if existing_data[0]['cloud_cover'] < 40 else (
                    "🌧️" if existing_data[0]['precipitation'] > 0 else "☁️"
                )
                if not (sunrise <= local_time <= sunset):  # If it's not daytime
                    if weather_symbol == "☀️":
                        weather_symbol = "🌙"

                map_data.append({
                    'city': city,
                    'lat': lat,
                    'lon': lon,
                    'temperature': existing_data[0]['temperature'],
                    'feels_like': existing_data[0]['feels_like'],
                    'humidity': existing_data[0]['humidity'],
                    'precipitation': existing_data[0]['precipitation'],
                    'cloud_cover': existing_data[0]['cloud_cover'],
                    'local_time': local_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'sunrise': sunrise.strftime('%Y-%m-%d %H:%M:%S'),
                    'sunset': sunset.strftime('%Y-%m-%d %H:%M:%S'),
                    'weather_symbol': weather_symbol
                })
            else:
                # Fetch weather data from the API
                weather_data = get_historical_hourly_weather(
                    api_key_weather, lat, lon, current_utc_time, current_utc_time - timedelta(hours=1), 1
                )

                if not weather_data.empty:
                    # Extract the first row of weather data
                    weather_row = weather_data.iloc[0]
                    temperature = weather_row['Temperature (°C)']
                    feels_like = weather_row['Feels Like (°C)']
                    humidity = float(weather_row['Humidity (%)'])
                    precipitation = weather_row['Precipitation (mm)']
                    cloud_cover = float(weather_row['Cloud Cover (%)'])

                    local_time = convert_utc_to_local(current_utc_time, lat, lon)

                    # Fetch sunrise and sunset times
                    sunrise, sunset = get_sunrise_sunset_times(lat, lon)

                    # Convert sunrise and sunset to local time
                    sunrise = convert_utc_to_local(sunrise, lat, lon)
                    sunset = convert_utc_to_local(sunset, lat, lon)       
                    # Determine the weather symbol
                    weather_symbol = "☀️" if cloud_cover < 40 else (
                        "🌧️" if precipitation > 0 else "☁️"
                    )
                    if not (sunrise <= local_time <= sunset):  # If it's not daytime
                        if weather_symbol == "☀️":
                            weather_symbol = "🌙"

                    # Convert UTC time to local time for the city

                    # Store the data in the database
                    db1.execute("""
                        INSERT INTO weather_map_data (
                            city, latitude, longitude, temperature, humidity, precipitation, utc_time, cloud_cover, feels_like, sunrise, sunset
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, city, lat, lon, temperature, humidity, precipitation, current_utc_time.strftime('%Y-%m-%d %H:%M:%S'), cloud_cover, feels_like, sunrise.strftime('%Y-%m-%d %H:%M:%S'), sunset.strftime('%Y-%m-%d %H:%M:%S'))

                    # Add the data to the map
                    map_data.append({
                        'city': city,
                        'lat': lat,
                        'lon': lon,
                        'temperature': temperature,
                        'feels_like': feels_like,
                        'humidity': humidity,
                        'precipitation': precipitation,
                        'cloud_cover': cloud_cover,
                        'local_time': local_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'sunrise': sunrise.strftime('%Y-%m-%d %H:%M:%S'),
                        'sunset': sunset.strftime('%Y-%m-%d %H:%M:%S'),
                        'weather_symbol': weather_symbol
                    })

        # Convert to DataFrame
        map_df = pd.DataFrame(map_data)

        # Generate the map
        import plotly.express as px
        fig = px.scatter_mapbox(
            map_df,
            lat='lat',
            lon='lon',
            hover_name='city',
            hover_data={
                'temperature': True,
                'feels_like': True,
                'humidity': True,
                'precipitation': True,
                'cloud_cover': True,
                'local_time': True,
                'sunrise': True,
                'sunset': True,
                'weather_symbol': True  # Add weather symbol to hover data
            },
            color='temperature',
            size='humidity',
            color_continuous_scale='Viridis',
            title=f"Weather Data for Cities in {country}"
        )
        fig.update_layout(mapbox_style='carto-positron', mapbox_zoom=4, mapbox_center={'lat': map_df['lat'].mean(), 'lon': map_df['lon'].mean()})
        fig.update_layout(margin={'r': 0, 't': 0, 'l': 0, 'b': 0})

        # Convert the map to HTML
        map_html = fig.to_html(full_html=False)

        return render_template('country_weather_map.html', map_html=map_html, country=country, last_updated=current_utc_time.strftime('%Y-%m-%d %H:%M:%S'))
    # Render the input form
    return render_template('country_weather.html')


@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')

@app.route('/live_weather', methods=['GET', 'POST'])
def live_weather():
    if request.method == 'POST':
        city = request.form['city'].strip()
        hours = int(request.form['hours'])

        # Get latitude and longitude for the given city
        lat, lon = get_lat_lon(city)
        if lat is None or lon is None:
            return render_template('live_weather.html', error=f"Could not fetch coordinates for {city}.")

        # Load the world_cities_lat_long file
        world_cities = pd.read_csv('data/world_cities_lat_long.csv')  # Ensure the file is in the correct path

        # Find nearby cities within a 500 km radius
        nearby_cities = []
        dist = 0
        map_data = [{'city': city, 'lat': lat, 'lon': lon, 'type': 'Requested City', 'distance': dist}]  # Add the requested city
        for _, row in world_cities.iterrows():
            other_city = row['city']
            other_country = row['country']
            other_lat = row['latitude']
            other_lon = row['longitude']
            if(city != other_city):
                distance = haversine(lat, lon, other_lat, other_lon)
                print(f"Other city is {other_city} and distance is {distance}")
                if distance <= 500:  # Restrict radius to 500 km
                    nearby_cities.append((other_city, other_country, round(distance, 2)))  # Add city, country, and distance
                    map_data.append({'city': other_city, 'lat': other_lat, 'lon': other_lon, 'type': 'Nearby City', 'distance': round(distance, 2)})

        nearby_cities = sorted(nearby_cities, key=lambda x: x[2])

        map_df = pd.DataFrame(map_data)
        fig = px.scatter_mapbox(
            map_df,
            lat='lat',
            lon='lon',
            hover_name='city',
            hover_data=['distance'],
            color='type',
            title=f"Requested City and Nearby Cities (500 km radius)",
            color_discrete_map={'Requested City': 'red', 'Nearby City': 'blue'},
            size=[10 if t == 'Requested City' else 5 for t in map_df['type']],  # Larger size for the requested city
        )
        fig.update_layout(mapbox_style='carto-positron', mapbox_zoom=4, mapbox_center={'lat': lat, 'lon': lon})
        fig.update_layout(margin={'r': 0, 't': 0, 'l': 0, 'b': 0})

        # Convert the map to HTML
        map_html = fig.to_html(full_html=False)

        end_time = datetime.now(pytz.utc).replace(minute=0, second=0, microsecond=0)
        start_time = end_time - timedelta(hours=hours)
        print(f"start time is {start_time}")
        print(f"end time is {end_time}")
        # Fetch hourly weather data from the API
        print(f"Fetching data for {city} from the API...")
        api_df = get_historical_hourly_weather(api_key_weather, lat, lon, end_time, start_time, hours)
        print(api_df)
        if api_df is None or api_df.empty:
            return render_template('live_weather.html', error="Could not fetch hourly weather data.")

        print(api_df)
        print("Nearby cities")
        print(map_data)
        # Render a Plotly table with the data
        import plotly.graph_objects as go
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=["Datetime (Local)", "Temperature (°C)", "Feels Like (°C)", "Precipitation (mm)",
                        "Wind Speed (kph)", "Wind Direction", "Cloud Cover (%)",
                        "Humidity (%)", "Pressure (mb)"],
                fill_color='paleturquoise',
                align='left'
            ),
            cells=dict(
                values=[
                    api_df['Datetime'].dt.strftime('%Y-%m-%d %H:%M'),
                    api_df['Temperature (°C)'],
                    api_df['Feels Like (°C)'],
                    api_df['Precipitation (mm)'],
                    api_df['Wind Speed (kph)'],
                    api_df['Wind Direction'],
                    api_df['Cloud Cover (%)'],
                    api_df['Humidity (%)'],
                    api_df['Pressure (mb)']
                ],
                fill_color='lavender',
                align='left'
            )
        )])

        table_html = fig.to_html(full_html=False)

        temperature_fig = px.line(
            api_df,
            x='Datetime',
            y=['Temperature (°C)', 'Feels Like (°C)'],
            labels={'value': 'Temperature (°C)', 'variable': 'Metric'},
            title=f"Temperature Trends for {city} (Last {hours} Hours)"
        )
        temperature_plot_html = temperature_fig.to_html(full_html=False)

        precipitation_fig = px.bar(
            api_df,
            x='Datetime',
            y='Precipitation (mm)',
            labels={'Precipitation (mm)': 'Precipitation (mm)'},
            title=f"Precipitation Trends for {city} (Last {hours} Hours)"
        )
        precipitation_plot_html = precipitation_fig.to_html(full_html=False)

        import plotly.graph_objects as go

        # Wind Rose Chart
        wind_rose_fig = go.Figure()
        wind_rose_fig.add_trace(go.Barpolar(
            r=api_df['Wind Speed (kph)'],
            theta=api_df['Wind Direction'],
            name='Wind Speed',
            marker_color='blue'
        ))
        wind_rose_fig.update_layout(
            title=f"Wind Speed and Direction for {city} (Last {hours} Hours)",
            polar=dict(
                angularaxis=dict(direction='clockwise')
            )
        )
        wind_rose_plot_html = wind_rose_fig.to_html(full_html=False)

        # Cloud Cover Line Plot
        cloud_cover_fig = px.line(
            api_df,
            x='Datetime',
            y='Cloud Cover (%)',
            labels={'Cloud Cover (%)': 'Cloud Cover (%)'},
            title=f"Cloud Cover Trends for {city} (Last {hours} Hours)"
        )
        cloud_cover_plot_html = cloud_cover_fig.to_html(full_html=False)

        # Humidity Area Plot
        humidity_fig = px.area(
            api_df,
            x='Datetime',
            y='Humidity (%)',
            labels={'Humidity (%)': 'Humidity (%)'},
            title=f"Humidity Trends for {city} (Last {hours} Hours)"
        )
        humidity_plot_html = humidity_fig.to_html(full_html=False)

        return render_template(
            'live_weather_plot.html',
            city=city,
            hours=hours,
            nearby_cities=nearby_cities,
            table_html=table_html,
            map_html=map_html,
            temperature_plot_html=temperature_plot_html,
            precipitation_plot_html=precipitation_plot_html,
            wind_rose_plot_html=wind_rose_plot_html,
            cloud_cover_plot_html=cloud_cover_plot_html,
            humidity_plot_html=humidity_plot_html
        )

    # Render the input form
    return render_template('live_weather.html')



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)