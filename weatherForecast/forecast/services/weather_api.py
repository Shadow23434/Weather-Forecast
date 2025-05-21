import requests
from .config import OPEN_WEATHER_API_KEY, CURRENT_WEATHER_URL, DEFAULT_CITY

# Dictionary mapping country codes to full country names
COUNTRY_NAMES = {
    "US": "United States",
    "GB": "United Kingdom",
    "FR": "France",
    "DE": "Germany",
    "IT": "Italy",
    "ES": "Spain",
    "JP": "Japan",
    "CN": "China",
    "RU": "Russia",
    "IN": "India",
    "BR": "Brazil",
    "CA": "Canada",
    "AU": "Australia",
    "MX": "Mexico",
    "KR": "South Korea",
    "ID": "Indonesia",
    "TR": "Turkey",
    "SA": "Saudi Arabia",
    "ZA": "South Africa",
    "AR": "Argentina",
    "TH": "Thailand",
    "EG": "Egypt",
    "VN": "Vietnam",
    "PH": "Philippines",
    "MY": "Malaysia",
    "PK": "Pakistan",
    "NG": "Nigeria",
    "NO": "Norway",
    "NZ": "New Zealand",
    "SE": "Sweden",
    "FI": "Finland",
    "DK": "Denmark",
    "AT": "Austria",
    "BE": "Belgium",
    "CH": "Switzerland",
    "NL": "Netherlands",
    "PT": "Portugal",
    "GR": "Greece",
    "IE": "Ireland",
    "SG": "Singapore",
    "IL": "Israel",
    "HK": "Hong Kong",
    "AE": "United Arab Emirates",
    "QA": "Qatar",
    "KW": "Kuwait",
    "OM": "Oman",
    "BH": "Bahrain",
}

# Get full country name from country code
def get_country_name(country_code):
    return COUNTRY_NAMES.get(country_code, country_code)

# Fetch Current Weather Data
def get_current_weather(city):
    if not city or city.strip() == "":
        return {
            'cod': 400,
            'message': "Nothing to geocode"
        }
    
    # Make sure city is lowercase and properly trimmed
    city = normalize_city_name(city)
        
    url = f"{CURRENT_WEATHER_URL}weather?q={city}&appid={OPEN_WEATHER_API_KEY}&units=metric"
    try:
        response = requests.get(url)
        data = response.json()
        
        cod = data['cod']
        if (cod != 200):
            return {
                'cod': cod,
                'message': data['message']
            }
        
        country_code = data['sys']['country']
        country_name = get_country_name(country_code)
        
        return {
            'cod': cod,
            'city': data['name'],
            'current_temp': round(data['main']['temp']),
            'feels_like': round(data['main']['feels_like']),
            'temp_min': round(data['main']['temp_min']),
            'temp_max': round(data['main']['temp_max']),
            'humidity': round(data['main']['humidity']),
            'description': data['weather'][0]['description'],
            'country': country_name,
            'country_code': country_code,
            'wind_gust_dir': data['wind']['deg'],
            'pressure': data['main']['pressure'],
            'wind_gust_speed': data['wind']['speed'],
            'clouds': data['clouds']['all'],
            'visibility': data['visibility'],
        }
    except requests.exceptions.RequestException:
        return {
            'cod': 500,
            'message': "Network error. Please check your internet connection."
        }
    except KeyError:
        return {
            'cod': 500,
            'message': "Invalid data received from weather service."
        }
    except Exception as e:
        return {
            'cod': 500,
            'message': f"An unexpected error occurred: {str(e)}"
        }

def normalize_city_name(city_name):
    """
    Standardize city names for consistent comparison without modifying original data
    
    Args:
        city_name (str): City name to normalize
        
    Returns:
        str: Normalized city name (lowercase, stripped)
    """
    if not city_name or not isinstance(city_name, str):
        return ""
    return city_name.lower().strip()

# Get City from IP Address
def get_city_from_ip(request):
    try:
        # Extract IP address from request
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0].strip()
        else:
            ip = request.META.get('REMOTE_ADDR', '')
        
        print(ip)
        if not ip:
            return normalize_city_name(DEFAULT_CITY)
        
        # Use ip-api.com for geolocation
        response = requests.get(f'http://ip-api.com/json/{ip}', timeout=5)
        if response.status_code != 200:
            return normalize_city_name(DEFAULT_CITY)
            
        data = response.json()
        if data.get('status') == 'success' and data.get('city'):
            # Return city name in lowercase for consistent matching
            return normalize_city_name(data['city'])
        else:
            return normalize_city_name(DEFAULT_CITY)
            
    except requests.exceptions.RequestException:
        return normalize_city_name(DEFAULT_CITY)
    except Exception as e:
        print(f"Error in get_city_from_ip: {str(e)}")
        return normalize_city_name(DEFAULT_CITY)

def get_weather_icon(description):
    """Map weather description to appropriate icon type"""
    description = description.lower()
    if 'rain' in description or 'shower' in description or 'drizzle' in description:
        return 'rain'
    elif 'cloud' in description:
        return 'cloudy'
    elif 'overcast' in description:
        return 'overcast'
    elif 'mist' in description or 'haze' in description or 'fog' in description:
        return 'mist'
    elif 'snow' in description or 'blizzard' in description:
        return 'snow'
    elif 'sleet' in description:
        return 'sleet'
    elif 'thunder' in description or 'storm' in description:
        return 'thunderstorm'
    elif 'clear' in description or 'sunny' in description:
        return 'clear-day'
    else:
        # Default fallback
        return 'clear-day' 