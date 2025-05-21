from django.shortcuts import render
from django.http import JsonResponse
from datetime import datetime
import os
import pandas as pd
from datetime import datetime, timedelta
from .forms import WeatherForm
import joblib
from django.core.cache import cache
import json

from .services import (
    get_current_weather, get_city_from_ip, get_weather_icon, normalize_city_name,
    forecast_temperature_from_csv, find_city_historical_data,
    get_capital_city, get_all_countries_and_capitals, fetch_capital_historical_data,
    fetch_all_capitals_historical_data
) 
from .services.config import DEFAULT_CITY, HISTORICAL_DATA_PATH, FORECAST_DAYS, FLAG_URL

# Add model storage paths
MODEL_STORAGE_PATH = os.path.join(HISTORICAL_DATA_PATH, 'models')
os.makedirs(MODEL_STORAGE_PATH, exist_ok=True)

def get_model_path(city, target_type):
    """Get path for stored model"""
    city_dir = os.path.join(MODEL_STORAGE_PATH, normalize_city_name(city))
    os.makedirs(city_dir, exist_ok=True)
    return os.path.join(city_dir, f'{target_type}_model.joblib')

def load_cached_model(city, target_type):
    """Load model from cache or storage"""
    cache_key = f'weather_model_{city}_{target_type}'
    
    # Try to get from cache first
    cached_model = cache.get(cache_key)
    if cached_model:
        return cached_model
    
    # If not in cache, try to load from storage
    model_path = get_model_path(city, target_type)
    if os.path.exists(model_path):
        try:
            model_data = joblib.load(model_path)
            # Cache for 6 hours
            cache.set(cache_key, model_data, 6 * 60 * 60)
            return model_data
        except Exception as e:
            print(f"Error loading model from storage: {str(e)}")
    
    return None

def save_model_to_storage(city, target_type, model_data):
    """Save model to storage and cache"""
    model_path = get_model_path(city, target_type)
    try:
        joblib.dump(model_data, model_path)
        cache_key = f'weather_model_{city}_{target_type}'
        cache.set(cache_key, model_data, 6 * 60 * 60)
    except Exception as e:
        print(f"Error saving model to storage: {str(e)}")

def weather_view(request):
    if request.method == 'POST':
        form = WeatherForm(request.POST)
        if form.is_valid():
            city = form.cleaned_data['city']
            city = normalize_city_name(city)
        else:
            city = get_city_from_ip(request)
    else:
        form = WeatherForm()
        city = request.GET.get('city', '')
        if not city:
            city = get_city_from_ip(request)
        else:
            city = normalize_city_name(city)
    
    if not city:
        city = normalize_city_name(DEFAULT_CITY)
    
    current_weather = get_current_weather(city)
    cod = current_weather.get('cod', 500)
    
    if cod != 200:
        context = {
            'cod': cod,
            'error_message': current_weather.get('message', 'Unknown error'),
            'city': city,
            'FLAG_URL': FLAG_URL,
            'form': form
        }
        return render(request, 'weather.html', context)

    icon_type = get_weather_icon(current_weather['description'])

    country_code = current_weather.get('country_code')
    historical_data_file = find_city_historical_data(
        HISTORICAL_DATA_PATH, city=city, country_code=country_code, default_file="weather.csv"
    )
    print("historical_data_file: ", historical_data_file)
    
    # Get current temperature
    current_temp = current_weather['current_temp']

    # Try to load cached models first
    min_model_data = load_cached_model(city, 'temp_min')
    max_model_data = load_cached_model(city, 'temp_max')

    if min_model_data and max_model_data:
        min_model, min_scaler, min_features, min_forecast = min_model_data
        max_model, max_scaler, max_features, max_forecast = max_model_data
    else:
        # If no cached models, train new ones
        min_model, min_scaler, min_features, min_forecast = forecast_temperature_from_csv(
            historical_data_file, target='temp_min', city=city
        )
        max_model, max_scaler, max_features, max_forecast = forecast_temperature_from_csv(
            historical_data_file, target='temp_max', city=city
        )

        # Save models if training was successful
        if min_model is not None and max_model is not None:
            save_model_to_storage(city, 'temp_min', (min_model, min_scaler, min_features, min_forecast))
            save_model_to_storage(city, 'temp_max', (max_model, max_scaler, max_features, max_forecast))

    # Format forecast dates
    forecast_dates = [datetime.now() + timedelta(days=i+1) for i in range(FORECAST_DAYS)]
    forecast_dates = [date.strftime("%B %d, %Y") for date in forecast_dates]
    
    # Create forecast days list
    forecast_days = []
    for i in range(FORECAST_DAYS):
        day_forecast = {
            'date': forecast_dates[i],
            'min_temp': min_forecast[i] if min_forecast else current_temp - 2,
            'max_temp': max_forecast[i] if max_forecast else current_temp + 2,
            'description': generate_weather_description(max_forecast[i] if max_forecast else current_temp + 2),
            'icon': get_weather_icon(generate_weather_description(max_forecast[i] if max_forecast else current_temp + 2))
        }
        forecast_days.append(day_forecast)

    # Calculate temperature percentage safely
    try:
        temp_diff = current_weather['temp_max'] - current_weather['temp_min']
        if temp_diff != 0:
            temp_percentage = ((current_weather['current_temp'] - current_weather['temp_min']) / temp_diff) * 100
        else:
            temp_percentage = 50  # Default to middle if min and max are the same
    except (KeyError, TypeError, ZeroDivisionError):
        temp_percentage = 50  # Default value if calculation fails

    context = {
        'cod': cod,
        'location': city,
        'current_temp': current_weather['current_temp'],
        'MinTemp': current_weather['temp_min'],
        'MaxTemp': current_weather['temp_max'],
        'feels_like': current_weather['feels_like'],
        'humidity': current_weather['humidity'],
        'clouds': current_weather['clouds'],
        'description': current_weather['description'],
        'icon_type': icon_type,
        'city': current_weather['city'],
        'country': current_weather['country'],
        'country_code': current_weather['country_code'],
        'time': datetime.now(),
        'date': datetime.now().strftime("%B %d, %Y"),
        'wind': current_weather['wind_gust_speed'],
        'pressure': current_weather['pressure'],
        'visibility': current_weather['visibility'],
        'forecast_days': forecast_days,
        'temp_percentage': temp_percentage,
        'FLAG_URL': FLAG_URL,
        'form': form,
    }

    return render(request, 'weather.html', context)

def generate_weather_description(max_temp):
    """
    Generate weather description based on maximum temperature.
    """
    # Handle None or invalid values
    if max_temp is None:
        return "Partly cloudy"  # Default description
    
    try:
        max_temp = float(max_temp)
    except (ValueError, TypeError):
        return "Partly cloudy"  # Default description for invalid values
        
    if max_temp > 33:
        return "Hot"
    elif max_temp > 28:
        return "Clear sky"
    elif max_temp > 24:
        return "Partly cloudy"
    elif max_temp > 20:
        return "Light rain"
    else:
        return "Rain"

# New view for historical capital city data
def capital_historical(request):
    country_code = request.GET.get('country', 'US').upper()
    # Get data source (NOAA or Open-Meteo, default to Open-Meteo)
    source = request.GET.get('source', 'open_meteo')
    
    try:
        # Set appropriate max days based on source
        max_days = 7300 if source.lower() == 'open_meteo' else 364
        # Get requested days, capped at max_days
        days = min(int(request.GET.get('days', 30)), max_days)
        
        # Get capital city for the country
        capital = get_capital_city(country_code)
        if not capital:
            return JsonResponse({
                'error': f'No capital city found for country code: {country_code}'
            }, status=404)
        
        # Fetch historical data
        data = fetch_capital_historical_data(country_code, days_back=days, source=source)
        
        if not data:
            return JsonResponse({
                'error': f'No historical data found for {capital}, {country_code} from {source} API'
            }, status=404)
        
        # Return the data
        return JsonResponse({
            'country_code': country_code,
            'capital': capital,
            'days': days,
            'source': source,
            'records_count': len(data),
            'data': data
        })
    except TypeError as e:
        return JsonResponse({
            'error': f'Data processing error: {str(e)}. There may be an issue with the data format from the {source} API.'
        }, status=500)
    except ValueError as e:
        return JsonResponse({
            'error': f'Invalid value: {str(e)}. Please check your input parameters.'
        }, status=400)
    except Exception as e:
        return JsonResponse({
            'error': f'Error fetching data: {str(e)}'
        }, status=500)

def capitals_list(request):
    """View to display a list of all available capital cities"""
    countries_and_capitals = get_all_countries_and_capitals()
    
    # Sort by country code
    countries_and_capitals.sort(key=lambda x: x[0])
    
    context = {
        'countries_and_capitals': countries_and_capitals,
        'count': len(countries_and_capitals)
    }
    
    return render(request, 'forecast/capitals.html', context)

def manual_historical_data(request):
    """View for manually fetching historical weather data for capital cities"""
    countries_and_capitals = get_all_countries_and_capitals()
    result_message = None
    file_path = None
    
    if request.method == 'POST':
        country_code = request.POST.get('country_code')
        # Get days value from POST, limited to maximum of 364 days for NOAA and 3650 for Open-Meteo
        days = int(request.POST.get('days', 3650))
        # Get data source (NOAA or Open-Meteo, default to Open-Meteo)
        source = request.POST.get('source', 'open_meteo')
        
        # Set max days based on source
        max_days = 3650 if source.lower() == 'open_meteo' else 365
        days = min(days, max_days)
        
        try:
            if country_code:
                # Get data for a specific country
                capital = get_capital_city(country_code)
                if capital:
                    data = fetch_capital_historical_data(country_code, days_back=days, source=source)
                    if data:
                        # Convert data to DataFrame and save
                        df = pd.DataFrame(data)
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        file_path = os.path.join(HISTORICAL_DATA_PATH, f"{country_code}_{source}_historical_{timestamp}.csv")
                        df.to_csv(file_path, index=False)
                        result_message = f"Successfully fetched data for {capital}, {country_code} for {days} days using {source.upper()} API. Saved to {file_path}."
                    else:
                        result_message = f"Could not fetch data for {capital}, {country_code} using {source.upper()} API. The API may be unavailable or the location may not have data."
                else:
                    result_message = f"No capital city found for country code: {country_code}"
            else:
                # Get data for all countries
                file_path = fetch_all_capitals_historical_data(days_back=days, source=source)
                if file_path:
                    result_message = f"Successfully fetched data for all capital cities for {days} days using {source.upper()} API. Saved to {file_path}."
                else:
                    result_message = f"Failed to fetch data for capital cities using {source.upper()} API. The API may be unavailable."
        except TypeError as e:
            result_message = f"Error processing data: {str(e)}. There may be an issue with the data format from the {source.upper()} API."
        except ValueError as e:
            result_message = f"Invalid value: {str(e)}. Please check your input parameters."
        except Exception as e:
            result_message = f"Error fetching data: {str(e)}"
    
    # Sort country list by code
    countries_and_capitals.sort(key=lambda x: x[0])
    
    context = {
        'countries_and_capitals': countries_and_capitals,
        'count': len(countries_and_capitals),
        'result_message': result_message,
        'file_path': file_path
    }
    
    return render(request, 'forecast/manual_historical.html', context)


    try:
        if len(df) < window + 1:
            return None, None
        df = prepare_lagged_features(df, col, window)
        df = df.dropna(subset=[col] + [f'{col}_lag_{i}' for i in range(1, window+1)])
        if len(df) < 2:
            return None, None
        X = df[[f'{col}_lag_{i}' for i in range(1, window+1)]]
        y = df[col]
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model, list(X.iloc[-1])
    except Exception as e:
        print(f"Error training model for {col}: {str(e)}")
        return None, None
