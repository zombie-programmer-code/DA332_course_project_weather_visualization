# Weather Vista

A Flask‑based web application delivering **real‑time**, **forecast**, and **historical** weather insights for hundreds of cities worldwide. It combines machine‑learning models, interactive Plotly/Mapbox maps, and a SQLite backend to provide rich visualizations and data‑driven forecasts.

---

## Features
- **Historical Trends**
  - Line and box plots(time frame is taken as input) for any combination of selected variables like Maximum Temperature, Minimum Temperature, Precipitation Hours for 30+ cities across India.
- **Weather Forecasts**  
  - Deep Learning‑powered predictions (max/min temp, rainfall in mm, maximum wind speed). Model used: Bidirectional LSTM with fully connected layers, with around 300000 parameters, trained for 30 epochs. 
  - Interactive tables, temperature, rainfall and wind speed predictions are displayed. Nearby cities are also highlighted on a map, with clickable links to get 3 day forecasts for any nearby city.
- **Live Weather**  
  - View live weather(upto last 24 hours) for any city in the world. 
  - Interactive weather table, temperature line plot, precipitation bar plot, wind rose chart, cloud cover trend, humidity area plot are displayed for the requested city. A map showing all the nearby cities(within 500 km) is displayed, and there are clickable buttons to fetch the live weather for any of the nearby cities.
- **Worldwide Live Weather Map**  
  - Produces a Scatter‑Mapbox of 50+ cities around the world with temperature, humidity, cloud cover, precipitation, local time, sunrise/sunset and weather emojis(clear, cloudy, rainy) visible on hovering over the city.
  - If latest data is already available in the database, it is rendered immediately else the API calls have to be made, which may take a few minutes. Weather data is updated every 1 hour.
- **Nationwide Live Weather Map**  
  - Produces a Scatter‑Mapbox of the major cities of the requested country showing temperature, humidity, cloud cover, precipitation, local time, sunrise/sunset and weather emojis(clear, cloudy, rainy) visible on hovering over the city.
- **Leaflet.js India weather visualization**
  - An interactive Leaflet map visualizing temperature distribution across India, with a heatmap overlay and geo-boundary outlines. Generated using Python’s folium library and rendered as a standalone HTML file.
- **Leaflet World Weather Map**
  - A standalone interactive Leaflet map displaying real-time temperature, cloud, wind, and precipitation overlays using OpenWeatherMap tiles. Includes city search with autocomplete and live temperature popups on map clicks. 
- **Weather analysis for India**
  - Monthly average statistics aggregated over the last 10 years(maximum temperature, minimum temperature and rainfall) over each region of India are displayed using interactive bar plots, with standard deviations highlighted as well.
- **Global Weather Analysis**
  - Visualizes global weather patterns using interactive Plotly charts: temperature trends, rainfall distributions, wind speeds, and more._-
  - Data is grouped by continent, country, and month, with insights like hottest cities and variable correlations.
<pre lang="markdown"> ## 📁 Project Structure ``` DA332_course_project_weather_visualization/ ├── app.py # Main Flask application ├── data/ # CSV data and precomputed stats │ ├── weather_india/ # Last 10 years daily weather data for 30+ cities in India │ ├── weather_world/ # Last 3 years daily weather data for 80+ cities worldwide │ ├── world_cities_lat_long.csv │ ├── world_cities_map.csv │ └── region_month_stats.csv ├── models/ # Pretrained ML models & scalers │ ├── weather_predictor_model.keras │ ├── weather_predictor_model_other.keras │ ├── x_scaler.pkl │ └── x_scaler_other.pkl ├── templates/ # Jinja2 HTML templates ├── static/ # Images ├── weather.db # SQLite: historical weather data ├── live_weather_map.db # SQLite: live map data ├── live_weather.db # SQLite: live weather data ├── city_lat_long.db # SQLite: city coordinates ├── README.md # Project documentation └── requirements.txt # Project dependencies ``` </pre>
## Tech Stack

### Frontend  
![Plotly](https://img.shields.io/badge/Plotly-Express%20%26%20Graph%20Objects-blue?logo=plotly&logoColor=white)  
![Leaflet](https://img.shields.io/badge/Leaflet.js-Interactive%20Maps-brightgreen?logo=leaflet&logoColor=white)  
![HTML](https://img.shields.io/badge/HTML5-E34F26?logo=html5&logoColor=white)  
![CSS](https://img.shields.io/badge/CSS3-1572B6?logo=css3&logoColor=white)  
![Bootstrap](https://img.shields.io/badge/Bootstrap-7952B3?logo=bootstrap&logoColor=white)  
![Jinja2](https://img.shields.io/badge/Jinja2-Template%20Engine-yellow?logo=jinja&logoColor=black)

### Backend  
![Flask](https://img.shields.io/badge/Flask-000000?logo=flask&logoColor=white)  
![SQLite](https://img.shields.io/badge/SQLite-003B57?logo=sqlite&logoColor=white)  
![CS50 SQL](https://img.shields.io/badge/CS50--SQL-SQL%20Wrapper-orange)

### Machine Learning  
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)  
![Keras](https://img.shields.io/badge/Keras-D00000?logo=keras&logoColor=white)  

### APIs  
![Open-Meteo](https://img.shields.io/badge/Open--Meteo-Archive%20Weather-blue)  
![WeatherAPI](https://img.shields.io/badge/WeatherAPI.com-Hourly%20Weather-lightblue)  
![Sunrise-Sunset](https://img.shields.io/badge/Sunrise--Sunset-Astronomy-lightgrey)  
![Maps.co](https://img.shields.io/badge/Maps.co-Geocoding-green)  
![OpenWeatherMap](https://img.shields.io/badge/OpenWeatherMap-Tile%20Layers-orange?logo=OpenWeatherMap&logoColor=white)


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
http://127.0.0.1:5000
```

