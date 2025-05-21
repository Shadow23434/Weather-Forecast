from datetime import datetime, timedelta
import pytz
from .config import DEFAULT_TIMEZONE, FORECAST_DAYS

def format_future_times(days=FORECAST_DAYS, timezone_str=DEFAULT_TIMEZONE):
    """
    Generate a list of formatted dates for future predictions
    
    Args:
        days: Number of future days to generate
        timezone_str: Timezone to use for time calculation
    
    Returns:
        List of formatted date strings (DD/MM format)
    """
    timezone = pytz.timezone(timezone_str)
    now = datetime.now(timezone)
    tomorrow = now + timedelta(days=1)
    
    # Return the next 'days' dates in DD/MM format
    return [(tomorrow + timedelta(days=i)).strftime("%d/%m") for i in range(days)]

def format_future_dates(days=FORECAST_DAYS, timezone_str=DEFAULT_TIMEZONE):
    """
    Generate a list of full date objects for future predictions
    
    Args:
        days: Number of future days to generate
        timezone_str: Timezone to use for time calculation
    
    Returns:
        List of datetime objects
    """
    timezone = pytz.timezone(timezone_str)
    now = datetime.now(timezone)
    tomorrow = now + timedelta(days=1)
    
    # Return the next 'days' dates as datetime objects
    return [tomorrow + timedelta(days=i) for i in range(days)]

def calculate_temp_percentage(min_temp, max_temp, current_temp):
    """
    Calculate the percentage for temperature slider
    
    Args:
        min_temp: Minimum temperature value
        max_temp: Maximum temperature value
        current_temp: Current temperature
        
    Returns:
        Integer percentage value (0-100)
    """
    # Ensure we don't divide by zero and handle edge cases
    if max_temp > min_temp:
        temp_percentage = int(((current_temp - min_temp) / (max_temp - min_temp)) * 100)
        # Clamp the value between 0 and 100
        temp_percentage = max(0, min(temp_percentage, 100))
    else:
        temp_percentage = 50  # Default to middle if min and max are equal
        
    return temp_percentage

def validate_date_range(start_date=None, end_date=None, days_back=None, max_days=364):
    """
    Validate and adjust date ranges for API requests to ensure they don't exceed limits.
    
    Args:
        start_date (str/datetime): Start date in 'YYYY-MM-DD' format or datetime object
        end_date (str/datetime): End date in 'YYYY-MM-DD' format or datetime object
        days_back (int): Days to go back from today (alternative to start_date)
        max_days (int): Maximum allowed days in range (default 364 for NOAA API)
        
    Returns:
        tuple: (start_date_str, end_date_str, is_valid)
    """
    # Convert strings to datetime objects if necessary
    if isinstance(start_date, str) and start_date:
        try:
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        except ValueError:
            return None, None, False
    
    if isinstance(end_date, str) and end_date:
        try:
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError:
            return None, None, False
    
    # Get current time and safe maximum end date (yesterday)
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    
    # If end_date is not provided, use yesterday to be safe
    if not end_date:
        end_date = yesterday
    
    # Ensure end_date is not in the future or today (to avoid API limits)
    if end_date >= today:
        end_date = yesterday
    
    # If start_date is not provided but days_back is, calculate start_date
    if not start_date and days_back:
        start_date = end_date - timedelta(days=days_back)
    
    # If we still don't have a start_date, use 30 days back as default
    if not start_date:
        start_date = end_date - timedelta(days=30)
    
    # Check the date range and adjust if needed
    date_range = (end_date - start_date).days
    
    if date_range <= 0:
        # Invalid range (start date is after end date)
        return None, None, False
    
    if date_range > max_days:
        # Adjust start_date to respect max_days limit
        start_date = end_date - timedelta(days=max_days)
    
    # Format dates as strings
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    return start_date_str, end_date_str, True 