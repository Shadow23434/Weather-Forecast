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

# Feature engineering parameters
WINDOW_SIZES = [3, 7, 14]  # For creating lag features
FEATURE_SELECTION_K = 10  # Number of features to select
OUTLIER_Z_THRESHOLD = 3  # Z-score threshold for outlier detection

# Testing parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_SPLITS = 5

# Time configuration
DEFAULT_TIMEZONE = 'Asia/Ho_Chi_Minh'
FORECAST_DAYS = 5

# Model weights for hybrid model
HYBRID_MODEL_WEIGHTS = {
    'ml_model': 0.6,
    'deep_model': 0.4
} 