# Capital Cities Weather Data Tool

This tool allows you to fetch historical weather data for the capital cities of all supported countries from OpenWeatherMap's historical data API.

## Features

- Fetches historical weather data for capitals of all supported countries
- Stores data in CSV format for easy analysis
- Can fetch maximum available historical data (up to 5 days back)
- Handles API rate limits and errors gracefully

## Requirements

- Python 3.x
- Django
- requests
- pandas

## Installation

1. Clone the repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Set up your OpenWeatherMap API key in the `.env` file:

```
OPEN_WEATHER_API_KEY=your_api_key_here
CURRENT_WEATHER_URL=https://api.openweathermap.org/data/2.5/weather
HISTORICAL_WEATHER_URL=https://history.openweathermap.org/data/2.5/history/city
FLAG_URL=https://flagcdn.com/w20
```

## Usage

### Command Line

You can fetch historical weather data for all capital cities using the Django management command:

```bash
# Fetch 5 days of historical data (maximum allowed by the API)
python manage.py fetch_capitals_weather --max

# Fetch a specific number of days (maximum 5)
python manage.py fetch_capitals_weather --days 3
```

### Programmatic Use

You can also use the tool programmatically in your Python code:

```python
from forecast.services.historical_weather import fetch_all_capitals_historical_data

# Fetch 5 days of historical data
file_path = fetch_all_capitals_historical_data(days_back=5)
print(f"Data saved to {file_path}")
```

## Data Format

The historical weather data is stored in CSV format with the following fields:

- city: Capital city name
- country_code: ISO country code
- country: Full country name
- date: Date in YYYY-MM-DD format
- time: Time in HH:MM:SS format
- timestamp: Unix timestamp
- temp: Temperature in Celsius
- feels_like: Feels-like temperature in Celsius
- pressure: Atmospheric pressure
- humidity: Humidity percentage
- temp_min: Minimum temperature
- temp_max: Maximum temperature
- visibility: Visibility in meters
- wind_speed: Wind speed in m/s
- wind_deg: Wind direction in degrees
- clouds: Cloudiness percentage
- weather_main: Main weather condition
- weather_description: Detailed weather description
- weather_icon: Weather icon code

## Limitations

- OpenWeatherMap's historical data API allows fetching data up to 5 days back.
- API rate limits may apply, and requests are throttled to avoid exceeding these limits.
- The tool requires an active internet connection and a valid OpenWeatherMap API key.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 