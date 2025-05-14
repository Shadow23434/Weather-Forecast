"""
Module for fetching and storing historical weather data for capital cities.
Uses NOAA Climate Data API and Open-Meteo API as data sources.
"""

import os
import csv
import datetime
import pandas as pd
from .capitals import get_all_countries_and_capitals, get_all_capitals_with_coordinates, get_capital_city, get_capital_coordinates
from .config import DATA_DIRECTORY
from .weather_api import get_country_name
from .noaa_weather import (
    fetch_noaa_capital_historical, fetch_all_capitals_noaa_data,
    convert_noaa_to_standard_format
)
from .open_meteo import (
    fetch_capital_open_meteo_historical, fetch_all_capitals_open_meteo_data
)
from .utils import validate_date_range

# Create data directory if it doesn't exist
os.makedirs(DATA_DIRECTORY, exist_ok=True)

# Source flags
SOURCE_NOAA = "noaa"
SOURCE_OPEN_METEO = "open_meteo"

def save_historical_data_to_csv(data, filename=None):
    """
    Save historical weather data to CSV file.
    
    Args:
        data (list): List of weather records
        filename (str, optional): Filename to save to. If None, a default name is used.
        
    Returns:
        str: Path to saved file
    """
    if not data:
        return None
    
    if not filename:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"historical_weather_{timestamp}.csv"
    
    file_path = os.path.join(DATA_DIRECTORY, filename)
    
    # Define CSV fields based on data structure
    fieldnames = list(data[0].keys())
    
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(file_path)
    
    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerows(data)
    
    return file_path

def fetch_all_capitals_historical_data(days_back=30, source=SOURCE_NOAA):
    """
    Fetch historical weather data for all capital cities.
    
    Args:
        days_back (int): Number of days back to fetch data for (default: 30)
        source (str): Data source to use ('noaa' or 'open_meteo')
        
    Returns:
        str: Path to saved CSV file
    """
    # Apply appropriate day limit based on the source
    if source.lower() == SOURCE_OPEN_METEO:
        # Open-Meteo supports up to 10 years of historical data
        max_days = 3650
    else:
        # NOAA API limit of 1 year
        max_days = 364
    
    # Ensure we don't exceed the API limit
    days_back = min(days_back, max_days)
    
    # Get all capital cities with their coordinates
    capitals_with_coordinates = get_all_capitals_with_coordinates()
    
    # Determine which API to use
    if source.lower() == SOURCE_OPEN_METEO:
        print(f"Using Open-Meteo API for fetching historical data for all capitals (up to {max_days} days)")
        # Fetch data using Open-Meteo
        data = fetch_all_capitals_open_meteo_data(capitals_with_coordinates, days_back)
        
        if not data:
            print("Open-Meteo API returned no data")
            return None
            
        # Save to CSV
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"openmeteo_capitals_historical_{timestamp}.csv"
        return save_historical_data_to_csv(data, filename)
    else:
        # Default to NOAA
        print(f"Using NOAA API for fetching historical data for all capitals (limited to {max_days} days)")
        
        # Fetch data using NOAA
        file_path = fetch_all_capitals_noaa_data(capitals_with_coordinates, days_back)
        
        if not file_path:
            print("NOAA API returned no data")
            return None
            
        return file_path

def fetch_capital_historical_data(country_code, days_back=30, start_date=None, end_date=None, source=SOURCE_NOAA):
    """
    Fetch historical weather data for a specific capital city.
    
    Args:
        country_code (str): Country code
        days_back (int): Number of days back to fetch data for. Default 30 days
        start_date (str): Optional specific start date in 'YYYY-MM-DD' format
        end_date (str): Optional specific end date in 'YYYY-MM-DD' format
        source (str): Data source to use ('noaa' or 'open_meteo')
        
    Returns:
        list: Historical weather data or empty list if error
    """
    # Get capital city and coordinates
    capital = get_capital_city(country_code)
    if not capital:
        print(f"No capital city found for country code: {country_code}")
        return []
    
    # Get coordinates for the capital
    lat, lon = get_capital_coordinates(capital, country_code)
    
    if not lat or not lon:
        print(f"Could not find coordinates for {capital}, {country_code}")
        return []
    
    # Set different max_days based on the data source
    if source.lower() == SOURCE_OPEN_METEO:
        # Open-Meteo supports up to 10 years of historical data
        max_days = 3650
    else:
        # NOAA API has a limit of 1 year
        max_days = 364
    
    # Validate and adjust date range with the appropriate max_days
    start_date_str, end_date_str, is_valid = validate_date_range(
        start_date=start_date,
        end_date=end_date,
        days_back=days_back,
        max_days=max_days
    )
    
    if not is_valid:
        print(f"Invalid date range provided for {country_code}")
        return []
    
    # Calculate days_back from the validated dates for the API call
    start_date_obj = datetime.datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date_obj = datetime.datetime.strptime(end_date_str, '%Y-%m-%d')
    days_back = (end_date_obj - start_date_obj).days
    
    # Determine which API to use
    if source.lower() == SOURCE_OPEN_METEO:
        print(f"Using Open-Meteo API for {capital}, {country_code} from {start_date_str} to {end_date_str}")
        
        # Fetch data from Open-Meteo
        standardized_data = fetch_capital_open_meteo_historical(
            country_code, 
            lat, 
            lon, 
            capital,
            days_back=days_back,
            start_date=start_date_str,
            end_date=end_date_str
        )
    else:
        # Default to NOAA
        print(f"Using NOAA API for {capital}, {country_code} from {start_date_str} to {end_date_str}")
        
        # Fetch data from NOAA
        noaa_data = fetch_noaa_capital_historical(country_code, lat, lon, days_back)
        
        if not noaa_data:
            print(f"NOAA API returned no data for {capital}, {country_code}")
            return []
        
        # Convert NOAA data to our standard format
        standardized_data = convert_noaa_to_standard_format(noaa_data)
    
    # Add city and country information if not already included
    for record in standardized_data:
        if "city" not in record or not record["city"]:
            record["city"] = capital
        if "country_code" not in record or not record["country_code"]:
            record["country_code"] = country_code
        if "country" not in record or not record["country"]:
            record["country"] = get_country_name(country_code)
    
    return standardized_data

def load_historical_data(file_path=None):
    """
    Load historical weather data from CSV file.
    
    Args:
        file_path (str, optional): Path to CSV file. If None, find most recent file.
        
    Returns:
        pandas.DataFrame: DataFrame with historical data
    """
    if not file_path:
        # Find the most recent CSV file in the data directory
        csv_files = [f for f in os.listdir(DATA_DIRECTORY) if f.endswith('.csv')]
        if not csv_files:
            return pd.DataFrame()
        
        # Sort by modification time (newest first)
        csv_files.sort(key=lambda x: os.path.getmtime(os.path.join(DATA_DIRECTORY, x)), reverse=True)
        file_path = os.path.join(DATA_DIRECTORY, csv_files[0])
    
    # Load data
    try:
        df = pd.read_csv(file_path)
        return df
    except:
        return pd.DataFrame()

def fetch_maximum_historical_data(source=SOURCE_NOAA):
    """
    Fetch the maximum possible historical weather data for all capital cities.
    
    Args:
        source (str): Data source to use ('noaa' or 'open_meteo')
    
    Returns:
        str: Path to saved CSV file
    """
    # Set the appropriate maximum days based on the data source
    if source.lower() == SOURCE_OPEN_METEO:
        # Open-Meteo supports up to 10 years (3650 days) of historical data
        max_days = 3650
    else:
        # NOAA API có giới hạn khoảng thời gian phải nhỏ hơn 1 năm (364 days)
        max_days = 364
        
    return fetch_all_capitals_historical_data(days_back=max_days, source=source) 
