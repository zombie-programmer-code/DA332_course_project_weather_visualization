# Weather Vista
Deployed mirror: [https://weather-vista.onrender.com](https://weather-vista.onrender.com)

A Flask‑based web application delivering **real‑time**, **forecast**, and **historical** weather insights for hundreds of cities worldwide. It combines machine‑learning models, interactive Plotly/Mapbox maps, and a SQLite backend to provide rich visualizations and data‑driven forecasts.

---

## Features
- **Historical Trends**
  - Explore multi-year weather trends and distributions across Indian cities using interactive line plots, box plots, and facet-based plots. Visualizations include city-wise temperature, rainfall, and wind patterns, along with monthly summaries across years.
- **Weather Forecasts**  
  - Deep Learning‑powered predictions (max/min temp, rainfall in mm, maximum wind speed). Model used: Bidirectional LSTM with fully connected layers, with around 300000 parameters, trained for 30 epochs. 
  - Interactive tables, temperature, rainfall and wind speed predictions are displayed. Nearby cities are also highlighted on a map, with clickable links to get 3 day forecasts for any nearby city.
- **Live Weather**  
  - Visualize the most recent hourly weather trends(upto 24 hours) for any city, including **temperature**, **humidity**, **wind speed**, and more.  
  - Get a detailed breakdown through **interactive line plots**, **bar charts**, **polar plots**, and **scatter visualizations**.  
Also view nearby cities(within 500 km) on an interactive **Mapbox map**, with dynamic markers and real-time weather symbols. There are clickable buttons to fetch the live weather for any of the nearby cities.
- **Worldwide Live Weather Map**  
  - Produces a Scatter‑Mapbox of 50+ cities around the world with temperature, humidity, cloud cover, precipitation, local time, sunrise/sunset and weather emojis(clear, cloudy, rainy) visible on hovering over the city.
  - If latest data is already available in the database, it is rendered immediately else the API calls have to be made, which may take a few minutes. Weather data is updated every 1 hour.
- **Nationwide Live Weather Map**  
  - Produces a Scatter‑Mapbox of the major cities of the requested country showing temperature, humidity, cloud cover, precipitation, local time, sunrise/sunset and weather emojis(clear, cloudy, rainy) visible on hovering over the city.
- **Leaflet World Weather Map**
  - A standalone interactive Leaflet map displaying real-time temperature, cloud, wind, and precipitation overlays using OpenWeatherMap tiles. Includes city search with autocomplete and live temperature popups on map clicks. 
- **Weather analysis for India**
  - Monthly average statistics aggregated over the last 10 years(maximum temperature, minimum temperature and rainfall) over each region of India are displayed using interactive bar plots, with standard deviations highlighted as well.
- **Global Weather Analysis**
  - Visualizes global weather patterns using interactive Plotly charts: temperature trends, rainfall distributions, wind speeds, and more._-
  - Data is grouped by continent, country, and month, with insights like hottest cities and variable correlations.
- **Additional Feature: Chatbot**
  - A chatbot capable of handling user queries and guiding him to the correct page according to his needs.
  
## 📁 Project Structure

```
DA332_course_project_weather_visualization/
├── app.py                         # Main Flask application  
├── data/                          # CSV data and precomputed stats  
│   ├── weather_india/             # Last 10 years daily weather data for 30+ cities in India  
│   ├── weather_world/             # Last 3 years daily weather data for 80+ cities worldwide  
│   ├── world_cities_lat_long.csv  
│   ├── world_cities_map.csv  
│   └── region_month_stats.csv  
├── models/                        # Pretrained ML models & scalers  
│   ├── weather_predictor_model.keras  
│   ├── weather_predictor_model_other.keras  
│   ├── x_scaler.pkl  
│   └── x_scaler_other.pkl  
├── templates/                     # Jinja2 HTML templates  
├── static/                        # Images  
├── weather.db                     # SQLite: historical weather data  
├── live_weather_map.db            # SQLite: live map data  
├── live_weather.db                # SQLite: live weather data  
├── city_lat_long.db               # SQLite: city coordinates  
├── README.md                      # Project documentation  
└── requirements.txt               # Project dependencies  
```


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

## Installation & Setup

1. **Clone the repository**  
   ```bash
   git clone https://github.com/zombie-programmer-code/DA332_course_project_weather_visualization
   cd DA332_course_project_weather_visualization
   ```

2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare data & models**  
   Make sure the following files are placed correctly:
   - Place all `*_weather.csv` files inside the `data/` directory.
   - Ensure these files are present in `data/`:
     - `combined_weather_data.csv`  
     - `world_cities_lat_long.csv`  
     - `world_cities_map.csv`  
     - `region_month_stats.csv`
   - Place the following in the `models/` directory:
     - `weather_predictor_model.keras`  
     - `weather_predictor_model_other.keras`  
     - `x_scaler.pkl`  
     - `x_scaler_other.pkl`

4. **Run the application**  
   ```bash
   flask run or python app.py
   ```

5. **Open in your browser**  
   [http://127.0.0.1:8000](http://127.0.0.1:8000)


