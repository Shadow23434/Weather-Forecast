"""
Module for fetching historical weather data from Open-Meteo API.
This provides an alternative source for historical weather data.
API Documentation: https://open-meteo.com/
"""

import requests
import datetime
import pandas as pd
from .config import DATA_DIRECTORY
from .utils import validate_date_range

def fetch_open_meteo_historical(latitude, longitude, start_date=None, end_date=None, days_back=30):
    """
    Fetch historical weather data from Open-Meteo API.
    
    Args:
        latitude (float): Latitude of the location
        longitude (float): Longitude of the location
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        days_back (int): Number of days to go back if start_date not provided
        
    Returns:
        dict: Weather data or None if error
    """
    try:
        # Validate and adjust date range (Open-Meteo supports up to 10 years of historical data)
        start_date_str, end_date_str, is_valid = validate_date_range(
            start_date=start_date,
            end_date=end_date,
            days_back=days_back,
            max_days=3650  # Open-Meteo supports up to 10 years of historical data
        )
        
        if not is_valid:
            print(f"Invalid date range provided for Open-Meteo API request")
            return None
            
        print(f"Using Open-Meteo API for coordinates ({latitude}, {longitude}) from {start_date_str} to {end_date_str}")
        
        # Double-check that end date is not in the future or today
        today = datetime.datetime.now().date()
        end_date_obj = datetime.datetime.strptime(end_date_str, '%Y-%m-%d').date()
        
        if end_date_obj >= today:
            # Use yesterday instead
            end_date_str = (today - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
            print(f"Adjusted end date to {end_date_str} to avoid API limitations")
        
        # Base URL for Open-Meteo historical data
        url = "https://archive-api.open-meteo.com/v1/archive"
        
        # Parameters for the API request
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date_str,
            "end_date": end_date_str,
            "daily": ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean", 
                     "precipitation_sum", "rain_sum", "snowfall_sum", 
                     "windspeed_10m_max", "windgusts_10m_max", "winddirection_10m_dominant"],
            "timezone": "auto"
        }
        
        try:
            # Make the API request with timeout
            response = requests.get(url, params=params, timeout=30)
            
            # Check if the request was successful
            if response.status_code == 200:
                data = response.json()
                # Validate the response structure
                if "daily" not in data:
                    print(f"Invalid data structure from Open-Meteo API: 'daily' field missing")
                    return None
                    
                # Check that we have data for the required fields
                required_fields = ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean"]
                for field in required_fields:
                    if field not in data["daily"] or not data["daily"][field]:
                        print(f"Required field '{field}' is missing or empty in Open-Meteo API response")
                        return None
                
                return data
            else:
                # Enhanced error reporting
                error_message = f"Error fetching Open-Meteo data: {response.status_code}"
                try:
                    error_detail = response.json()
                    if isinstance(error_detail, dict) and "reason" in error_detail:
                        error_message += f", {error_detail['reason']}"
                except:
                    error_message += f", {response.text}"
                
                print(error_message)
                return None
        except requests.exceptions.Timeout:
            print(f"Timeout while connecting to Open-Meteo API")
            return None
        except requests.exceptions.ConnectionError:
            print(f"Connection error with Open-Meteo API")
            return None
        except Exception as e:
            print(f"Exception fetching Open-Meteo data: {str(e)}")
            return None
    except Exception as outer_e:
        print(f"Unexpected error in fetch_open_meteo_historical: {str(outer_e)}")
        return None

def convert_open_meteo_to_standard_format(data, city_name=None, country_code=None):
    """
    Convert Open-Meteo data to the standard format used in the application.
    
    Args:
        data (dict): Raw Open-Meteo data
        city_name (str): City name to include in the records
        country_code (str): Country code to include in the records
        
    Returns:
        list: Standardized weather records
    """
    if not data or "daily" not in data:
        return []
    
    standard_data = []
    
    daily = data["daily"]
    dates = daily.get("time", [])
    
    for i, date in enumerate(dates):
        # Extract weather data for this date
        try:
            temp_max = daily["temperature_2m_max"][i]
            temp_min = daily["temperature_2m_min"][i]
            temp = daily["temperature_2m_mean"][i]
            precipitation = daily.get("precipitation_sum", [])[i] if i < len(daily.get("precipitation_sum", [])) else None
            wind_speed = daily.get("windspeed_10m_max", [])[i] if i < len(daily.get("windspeed_10m_max", [])) else None
            wind_deg = daily.get("winddirection_10m_dominant", [])[i] if i < len(daily.get("winddirection_10m_dominant", [])) else None
            
            # Handle None values for precipitation
            has_rain = False
            if precipitation is not None and precipitation > 0:
                has_rain = True
            
            # Create standardized record
            record = {
                "city": city_name or "Unknown",
                "country_code": country_code or "",
                "date": date,
                "time": "12:00:00",  # Open-Meteo daily data, so use noon as default time
                "timestamp": int(datetime.datetime.strptime(f"{date} 12:00:00", "%Y-%m-%d %H:%M:%S").timestamp()),
                "temp": temp,
                "temp_min": temp_min,
                "temp_max": temp_max, # Open-Meteo doesn't provide pressure in basic dataset
                "wind_speed": wind_speed,
                "wind_deg": wind_deg,
                # "weather_main": "Rain" if has_rain else "Clear",
                # "weather_description": "Precipitation" if has_rain else "Clear sky",
                "precipitation": precipitation or 0,  # Default to 0 if None
                "snow": daily.get("snowfall_sum", [])[i] if i < len(daily.get("snowfall_sum", [])) else None,
            }
            
            standard_data.append(record)
        except (IndexError, KeyError) as e:
            print(f"Error processing Open-Meteo data for date {date}: {e}")
            continue
    
    return standard_data

def fetch_capital_open_meteo_historical(country_code, capital_lat, capital_lon, capital_name, days_back=30, start_date=None, end_date=None):
    """
    Fetch historical weather data for a capital city using Open-Meteo API.
    
    Args:
        country_code (str): Country code
        capital_lat (float): Capital city latitude
        capital_lon (float): Capital city longitude
        capital_name (str): Capital city name
        days_back (int): Number of days to go back
        start_date (str): Optional specific start date in 'YYYY-MM-DD' format
        end_date (str): Optional specific end date in 'YYYY-MM-DD' format
        
    Returns:
        list: Weather records in standard format or empty list if error
    """
    # Log the request
    print(f"Fetching historical data for {capital_name}, {country_code} using Open-Meteo API...")
    
    # Calculate safe end date (yesterday)
    if end_date is None:
        yesterday = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
        end_date = yesterday
    
    # Fetch data from Open-Meteo
    raw_data = fetch_open_meteo_historical(
        capital_lat, 
        capital_lon, 
        start_date=start_date,
        end_date=end_date,
        days_back=days_back
    )
    
    if not raw_data:
        print(f"Open-Meteo API returned no data for {capital_name}, {country_code}")
        return []
    
    # Convert to standard format
    standardized_data = convert_open_meteo_to_standard_format(
        raw_data, 
        city_name=capital_name, 
        country_code=country_code
    )
    
    print(f"Successfully fetched {len(standardized_data)} days of data for {capital_name}, {country_code}")
    return standardized_data

def fetch_all_capitals_open_meteo_data(capitals_with_coordinates, days_back=30):
    """
    Fetch Open-Meteo historical data for multiple capital cities.
    
    Args:
        capitals_with_coordinates (list): List of (country_code, capital, lat, lon) tuples
        days_back (int): Number of days to go back
        
    Returns:
        list: Combined weather data for all capitals
    """
    all_data = []
    
    for country_code, capital, lat, lon in capitals_with_coordinates:
        print(f"Fetching Open-Meteo data for {capital}, {country_code}...")
        
        data = fetch_capital_open_meteo_historical(
            country_code, 
            lat, 
            lon, 
            capital, 
            days_back=days_back
        )
        
        if data:
            all_data.extend(data)
            print(f"Retrieved {len(data)} records for {capital}")
        else:
            print(f"No data retrieved for {capital}")
    
    print(f"Completed fetching Open-Meteo data for {len(capitals_with_coordinates)} capitals, got {len(all_data)} total records")
    
    return all_data 