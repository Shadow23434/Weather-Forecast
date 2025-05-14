from django.conf import settings
import os

# API configuration
API_KEY = settings.API_KEY
BASE_URL = settings.BASE_URL
FLAG_URL = settings.FLAG_URL

# Default locations
DEFAULT_CITY = 'Ho Chi Minh City'

# File paths
HISTORICAL_DATA_PATH = os.path.join('C:\\Projects\\Python\\WeatherForecast\\weather.csv')

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
FORECAST_HOURS = 5 