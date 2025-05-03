# 🌦️ Weather Insight Platform

A Flask‑based web application delivering **real‑time**, **forecast**, and **historical** weather insights for hundreds of cities worldwide. It combines machine‑learning models, interactive Plotly/Mapbox maps, and a SQLite backend to provide rich visualizations and data‑driven forecasts.

---

## 🚀 Features

- **Live Weather Map**  
  - Scatter‑Mapbox of 100+ cities showing temperature, humidity, cloud cover, precipitation, local time, sunrise/sunset and weather emojis  
  - Highlights requested city and nearby cities (≤ 500 km)  

- **7‑Day Forecasts**  
  - ML‑powered predictions (max/min temp, rainfall in mm & category, wind speed)  
  - Separate Keras models for Indian vs. international cities  
  - Interactive tables, line charts, bar plots  

- **Historical Trends**  
  - Hourly data for past N hours: temperature, feels‑like, precipitation, wind, cloud cover, humidity, pressure  
  - Line, area, bar, box and wind‑rose charts  

- **Global Analytics Dashboard**  
  - Monthly average/max/min temperatures by continent & country  
  - Rainfall choropleth by country  
  - Hottest & coldest city per continent per month (faceted bar plots)  
  - Correlation heatmaps of weather variables  

---

## 🗂️ Repository Structure

DA332_course_project_weather_visualization/
├── app.py # Main Flask application
├── data/ # CSV data and precomputed stats
│ ├── <City>_weather.csv # Raw daily weather per city
│ ├── combined_weather_data.csv
│ ├── world_cities_lat_long.csv
│ ├── world_cities_map.csv
│ └── region_month_stats.csv
├── models/ # Pretrained ML models & scalers
│ ├── weather_predictor_model.keras
│ ├── weather_predictor_model_other.keras
│ ├── x_scaler.pkl
│ └── x_scaler_other.pkl
├── templates/ # Jinja2 HTML templates
├── static/ # CSS, JS, images
├── weather.db # SQLite: historical weather data
├── live_weather_map.db # SQLite: live map data
├── city_lat_long.db # SQLite: city coordinates
└── README.md # Project documentation

---

## ⚙️ Tech Stack

- **Frontend**: Plotly Express & Graph Objects, HTML, Jinja2, Bootstrap  
- **Backend**: Flask, SQLite (via CS50 SQL wrapper)  
- **ML**: Keras (TensorFlow), Scikit‑learn (RandomForestRegressor, StandardScaler)  
- **APIs**:
  - Open‑Meteo Archive API  
  - WeatherAPI.com (hourly history)  
  - Sunrise‑Sunset API  
  - Maps.co Geocoding API  

---

## ▶️ Installation & Setup

1. **Clone the repository**  
   ```bash
   git clone https://github.com/zombie-programmer-code/DA332_course_project_weather_visualization
   cd DA332_course_project_weather_visualization
   ```
2. **Installation**
Install dependencies
```bash
  pip install -r requirements.txt
```
3. Prepare data & models

Place all *_weather.csv files in data/ directory
Ensure combined_weather_data.csv, mapping CSVs, and region_month_stats.csv are present
Put Keras model files and scaler .pkl files in models/ directory

Run the application
```bash
flask run
```
Open in browser:
```bash
http://localhost:8000/
```
4. Features

Home: Overview and navigation dashboard
Live Weather Map: Real-time city weather with interactive map visualization
Forecasts: Enter a city and days ahead to see ML-powered weather forecasts
Historical Trends: Select cities, years, and variables for trend analysis and comparison
Global Analysis: View precomputed global charts and climate extremes

5. Technologies Used

Python
Flask
Keras/TensorFlow
Pandas
Plotly

6. Requirements

Python 3.8+
Internet connection for live weather data
Modern web browser with JavaScript enabled

