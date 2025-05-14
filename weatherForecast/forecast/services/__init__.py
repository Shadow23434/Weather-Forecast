# This file makes the services directory a Python package.
# Import key functions to simplify imports in other modules
from .weather_api import get_current_weather, get_city_from_ip, get_weather_icon, normalize_city_name
from .ml_predictions import (
    read_historical_data, prepare_data, train_rain_model, 
    prepare_regression_data, train_regression_model, enrich_historical_data,
    predict_future, map_wind_direction, find_city_historical_data
)
from .utils import format_future_times, calculate_temp_percentage
from .capitals import get_capital_city, get_all_capital_cities, get_all_countries_and_capitals, get_capital_coordinates, get_all_capitals_with_coordinates
from .historical_weather import (
    fetch_maximum_historical_data, fetch_all_capitals_historical_data,
    fetch_capital_historical_data, load_historical_data, save_historical_data_to_csv,
    SOURCE_NOAA, SOURCE_OPEN_METEO
)
from .noaa_weather import (
    fetch_noaa_historical_data, fetch_noaa_capital_historical, 
    fetch_all_capitals_noaa_data, convert_noaa_to_standard_format,
    get_noaa_stations_by_location, get_country_stations, process_noaa_data
)
from .open_meteo import (
    fetch_open_meteo_historical, fetch_capital_open_meteo_historical,
    fetch_all_capitals_open_meteo_data, convert_open_meteo_to_standard_format
) 