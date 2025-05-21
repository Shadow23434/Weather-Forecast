"""
Module for fetching historical weather data from NOAA Climate Data API.
This is used as the primary data source for historical weather information.
"""

import os
import csv
import time
import json
import math
import datetime
import requests
import pandas as pd
from .config import DATA_DIRECTORY, NOAA_API_TOKEN, NOAA_API_BASE_URL, NOAA_DATASET_ID


# Create data directory if it doesn't exist
os.makedirs(DATA_DIRECTORY, exist_ok=True)

def get_noaa_stations_by_location(latitude, longitude, radius=25, limit=10):
    """
    Find NOAA weather stations near a specified location.
    
    Args:
        latitude (float): Latitude of the location
        longitude (float): Longitude of the location
        radius (int): Radius in kilometers to search for stations
        limit (int): Maximum number of stations to return
        
    Returns:
        list: A list of station dictionaries with id, name, and location
    """
    headers = {"token": NOAA_API_TOKEN}
    params = {
        "extent": f"{latitude-radius/111.32},{longitude-radius/(111.32*math.cos(latitude*0.0174533))},{latitude+radius/111.32},{longitude+radius/(111.32*math.cos(latitude*0.0174533))}",
        "limit": limit
    }
    
    url = f"{NOAA_API_BASE_URL}/stations"
    
    try:
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            return data.get("results", [])
        else:
            print(f"Error getting NOAA stations: {response.status_code}, {response.text}")
            return []
    except Exception as e:
        print(f"Exception getting NOAA stations: {e}")
        return []

def get_country_stations(country_code, limit=100):
    """
    Get NOAA weather stations for a specific country.
    
    Args:
        country_code (str): Two-letter country code
        limit (int): Maximum number of stations to return
        
    Returns:
        list: A list of station dictionaries
    """
    headers = {"token": NOAA_API_TOKEN}
    params = {
        "locationid": f"FIPS:{country_code}",
        "limit": limit,
        "sortfield": "name"
    }
    
    url = f"{NOAA_API_BASE_URL}/stations"
    
    try:
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            return data.get("results", [])
        else:
            print(f"Error getting stations for {country_code}: {response.status_code}, {response.text}")
            return []
    except Exception as e:
        print(f"Exception getting stations for {country_code}: {e}")
        return []

def fetch_noaa_historical_data(station_id, start_date, end_date, data_types=None):
    """
    Fetch historical weather data from NOAA for a specific station.
    
    Args:
        station_id (str): NOAA station ID
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        data_types (list): List of data types to fetch. If None, fetches common weather data.
        
    Returns:
        dict: Weather data or error message
    """
    # Check and adjust date range - NOAA API requires less than 1 year range
    start_date_obj = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_date_obj = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    
    # Calculate number of days between dates
    date_range = (end_date_obj - start_date_obj).days
    
    # If date range is greater than 364 days (almost 1 year), adjust start_date
    if date_range > 364:
        print(f"Adjusting date range from {date_range} days to 364 days (NOAA API limit)")
        start_date_obj = end_date_obj - datetime.timedelta(days=364)
        start_date = start_date_obj.strftime('%Y-%m-%d')
    
    headers = {"token": NOAA_API_TOKEN}
    
    # Default to common weather data types if none provided
    if data_types is None:
        data_types = ["TMAX", "TMIN", "PRCP", "SNOW", "SNWD", "AWND"]
    
    params = {
        "datasetid": NOAA_DATASET_ID,
        "stationid": station_id,
        "startdate": start_date,
        "enddate": end_date,
        "datatypeid": ",".join(data_types),
        "units": "metric",
        "limit": 1000  # Maximum allowed by API
    }
    
    url = f"{NOAA_API_BASE_URL}/data"
    
    all_results = []
    
    try:
        # Handle pagination as NOAA API limits results per request
        offset = 1
        more_data = True
        max_retries = 3
        retry_count = 0
        
        while more_data and retry_count < max_retries:
            params["offset"] = offset
            
            try:
                # Add timeout to prevent hanging on slow responses
                response = requests.get(url, headers=headers, params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    
                    if results:
                        all_results.extend(results)
                        
                        # Check if there might be more data
                        if len(results) == 1000:  # Hit the limit
                            offset += 1000
                        else:
                            more_data = False
                    else:
                        more_data = False
                    
                    # Reset retry counter on success
                    retry_count = 0
                    
                elif response.status_code == 429:  # Rate limit exceeded
                    print(f"Rate limit exceeded. Waiting before retry...")
                    retry_count += 1
                    time.sleep(5)  # Wait longer before retry
                elif response.status_code >= 500:  # Server error
                    print(f"NOAA server error: {response.status_code}. Retrying...")
                    retry_count += 1
                    time.sleep(2)
                else:
                    print(f"Error fetching NOAA data: {response.status_code}, {response.text}")
                    return {"error": f"API error: {response.status_code}", "message": response.text}
            
            except requests.exceptions.Timeout:
                print(f"Request to NOAA API timed out. Retrying... ({retry_count + 1}/{max_retries})")
                retry_count += 1
                time.sleep(2)
                
            except requests.exceptions.RequestException as e:
                print(f"Network error when connecting to NOAA API: {e}")
                return {"error": "Network error", "message": str(e)}
            
            # Respect API rate limits
            time.sleep(0.5)
            
        return {"results": all_results}
    except Exception as e:
        print(f"Exception fetching NOAA data: {e}")
        return {"error": str(e)}

def process_noaa_data(noaa_data):
    """
    Process raw NOAA API data into a structured format.
    
    Args:
        noaa_data (dict): Raw data from NOAA API
        
    Returns:
        list: Processed weather records
    """
    if "error" in noaa_data:
        return []
    
    results = noaa_data.get("results", [])
    if not results:
        return []
    
    # Group data by date
    data_by_date = {}
    
    for item in results:
        date = item.get("date")
        if date:
            date = date.split("T")[0]  # Remove time part
            
            if date not in data_by_date:
                data_by_date[date] = {}
            
            data_type = item.get("datatype")
            value = item.get("value")
            
            if data_type and value is not None:
                data_by_date[date][data_type] = value
    
    # Convert to our standard format
    formatted_data = []
    
    for date, values in data_by_date.items():
        # Convert temperature from tenths of degrees C to degrees C
        temp_max = float(values.get("TMAX", 0)) / 10 if "TMAX" in values else None
        temp_min = float(values.get("TMIN", 0)) / 10 if "TMIN" in values else None
        
        # Calculate average temperature if max and min are available
        temp = None
        if temp_max is not None and temp_min is not None:
            temp = (temp_max + temp_min) / 2
        
        # Convert precipitation from tenths of mm to mm
        precipitation = float(values.get("PRCP", 0)) / 10 if "PRCP" in values else 0
        
        # Wind speed is in meters per second
        wind_speed = float(values.get("AWND", 0)) if "AWND" in values else None
        
        record = {
            "date": date,
            "time": "12:00:00",  # NOAA data is daily, so use noon as default time
            "temp": temp,
            "temp_min": temp_min,
            "temp_max": temp_max,
            "precipitation": precipitation,
            "wind_speed": wind_speed,
            "snow": values.get("SNOW"),
            "snow_depth": values.get("SNWD")
        }
        
        formatted_data.append(record)
    
    return formatted_data

def save_noaa_data_to_csv(data, country_code, filename=None):
    """
    Save processed NOAA data to CSV.
    
    Args:
        data (list): Processed weather records
        country_code (str): Country code
        filename (str, optional): Custom filename
        
    Returns:
        str: Path to saved file or None if error
    """
    if not data:
        return None
    
    if not filename:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"noaa_historical_{country_code}_{timestamp}.csv"
    
    file_path = os.path.join(DATA_DIRECTORY, filename)
    
    try:
        # Define CSV fields based on data structure
        fieldnames = list(data[0].keys())
        
        with open(file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        
        return file_path
    except Exception as e:
        print(f"Error saving NOAA data to CSV: {e}")
        return None

def fetch_noaa_capital_historical(country_code, capital_lat, capital_lon, days_back=30):
    """
    Fetch historical weather data for a capital city using NOAA API.
    
    Args:
        country_code (str): Country code
        capital_lat (float): Capital city latitude
        capital_lon (float): Capital city longitude
        days_back (int): Number of days to go back
        
    Returns:
        list: Weather records or empty list if error
    """
    # Calculate date range
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.datetime.now() - datetime.timedelta(days=days_back)).strftime('%Y-%m-%d')
    
    # Find nearby stations
    stations = get_noaa_stations_by_location(capital_lat, capital_lon)
    
    if not stations:
        print(f"No NOAA stations found near {capital_lat}, {capital_lon}")
        return []
    
    # Use the first station found
    station = stations[0]
    station_id = station.get("id")
    
    if not station_id:
        print("Invalid station data")
        return []
    
    # Fetch data
    noaa_data = fetch_noaa_historical_data(station_id, start_date, end_date)
    
    # Process data
    processed_data = process_noaa_data(noaa_data)
    
    # Add location information to each record
    for record in processed_data:
        record["station_id"] = station_id
        record["station_name"] = station.get("name")
        record["country_code"] = country_code
    
    return processed_data

def fetch_all_capitals_noaa_data(capitals_with_coordinates, days_back=30):
    """
    Fetch NOAA historical data for multiple capital cities.
    
    Args:
        capitals_with_coordinates (list): List of (country_code, capital, lat, lon) tuples
        days_back (int): Number of days to go back
        
    Returns:
        str: Path to saved CSV file
    """
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"noaa_capitals_historical_{timestamp}.csv"
    file_path = os.path.join(DATA_DIRECTORY, filename)
    
    all_data = []
    success_count = 0
    
    for country_code, capital, lat, lon in capitals_with_coordinates:
        print(f"Fetching NOAA data for {capital}, {country_code}...")
        
        data = fetch_noaa_capital_historical(country_code, lat, lon, days_back)
        
        if data:
            # Add capital city name to each record
            for record in data:
                record["city"] = capital
            
            all_data.extend(data)
            success_count += 1
            
            # Save incrementally to avoid losing data if there's an error
            if all_data:
                fieldnames = list(all_data[0].keys())
                
                # Check if file exists
                file_exists = os.path.isfile(file_path)
                
                with open(file_path, mode='a', newline='', encoding='utf-8') as file:
                    writer = csv.DictWriter(file, fieldnames=fieldnames)
                    
                    if not file_exists:
                        writer.writeheader()
                    
                    writer.writerows(data)
        
        # Respect API rate limits
        time.sleep(1)
    
    print(f"Completed fetching NOAA data for {success_count} capitals")
    
    return file_path if all_data else None

def convert_noaa_to_standard_format(noaa_data):
    """
    Convert NOAA data to the standard format used in the application.
    
    Args:
        noaa_data (list): NOAA weather records
        
    Returns:
        list: Standardized weather records
    """
    standard_data = []
    
    for record in noaa_data:
        # Convert temperature from Celsius to the standard format
        temp = record.get("temp")
        temp_min = record.get("temp_min")
        temp_max = record.get("temp_max")
        
        # Handle precipitation (rainfall)
        rainfall = record.get("precipitation", 0)
        
        # Handle wind speed
        wind_speed = record.get("wind_speed")
        
        # Create standardized record
        std_record = {
            "city": record.get("city", "Unknown"),
            "country_code": record.get("country_code", ""),
            "date": record.get("date"),
            "time": record.get("time", "12:00:00"),
            "timestamp": int(datetime.datetime.strptime(f"{record.get('date')} {record.get('time', '12:00:00')}", "%Y-%m-%d %H:%M:%S").timestamp()),
            "temp": temp,
            "temp_min": temp_min,
            "temp_max": temp_max,
            "humidity": None,  # NOAA doesn't provide humidity in basic dataset
            "pressure": None,  # NOAA doesn't provide pressure in basic dataset
            "wind_speed": wind_speed,
            "wind_deg": None,  # NOAA doesn't provide wind direction in basic dataset
            "clouds": None,  # NOAA doesn't provide cloud coverage in basic dataset
            "weather_main": "Rain" if rainfall > 0 else "Clear",
            "weather_description": "Precipitation" if rainfall > 0 else "Clear sky",
        }
        
        standard_data.append(std_record)
    
    return standard_data 