from datetime import datetime, timedelta
import pytz
from .config import DEFAULT_TIMEZONE, FORECAST_HOURS

def format_future_times(hours=FORECAST_HOURS, timezone_str=DEFAULT_TIMEZONE):
    """
    Generate a list of formatted times for future predictions
    
    Args:
        hours: Number of future hours to generate
        timezone_str: Timezone to use for time calculation
    
    Returns:
        List of formatted time strings (HH:00 format)
    """
    timezone = pytz.timezone(timezone_str)
    now = datetime.now(timezone)
    next_hour = now + timedelta(hours=1)
    next_hour = next_hour.replace(minute=0, second=0, microsecond=0)

    return [(next_hour + timedelta(hours=i)).strftime("%H:00") for i in range(hours)]

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