from django.conf import settings
import os

# API configuration
OPEN_WEATHER_API_KEY = settings.OPEN_WEATHER_API_KEY
CURRENT_WEATHER_URL = settings.CURRENT_WEATHER_URL
HISTORICAL_WEATHER_URL = settings.HISTORICAL_WEATHER_URL
FLAG_URL = settings.FLAG_URL
NOAA_API_TOKEN=settings.NOAA_API_TOKEN
NOAA_API_BASE_URL=settings.NOAA_API_BASE_URL
NOAA_DATASET_ID =settings.NOAA_DATASET_ID

# Data storage configuration
DATA_DIRECTORY = os.path.join('C:\\Projects\\Python\\WeatherForecast\\data')

# Default locations
DEFAULT_CITY = 'Ha noi'

# File paths
HISTORICAL_DATA_PATH = DATA_DIRECTORY

# Model parameters
RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'random_state': 42
}

# Testing parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Time configuration
DEFAULT_TIMEZONE = 'Asia/Ho_Chi_Minh'
FORECAST_DAYS = 5 