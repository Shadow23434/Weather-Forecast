# This file makes the services directory a Python package.
# Import key functions to simplify imports in other modules
from .weather_api import get_current_weather, get_city_from_ip, get_weather_icon
from .ml_predictions import (
    read_historical_data, prepare_data, train_rain_model, 
    prepare_regression_data, train_regression_model, 
    predict_future, map_wind_direction
)
from .utils import format_future_times, calculate_temp_percentage 