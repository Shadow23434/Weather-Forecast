from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from datetime import datetime
import os
import pandas as pd
import json
from datetime import datetime, timedelta
from django.conf import settings

# Import form
from .forms import WeatherForm

# Import services
from .services import (
    get_current_weather, get_city_from_ip, get_weather_icon, normalize_city_name,
    read_historical_data, prepare_data, train_rain_model,
    prepare_regression_data, train_regression_model,
    predict_future, map_wind_direction,
    format_future_times, calculate_temp_percentage,
    enrich_historical_data,
    get_capital_city, get_all_countries_and_capitals, fetch_capital_historical_data,
    fetch_all_capitals_historical_data, find_city_historical_data,
    get_all_capitals_with_coordinates, SOURCE_NOAA, SOURCE_OPEN_METEO
)
from .services.config import DEFAULT_CITY, HISTORICAL_DATA_PATH, FORECAST_DAYS, FLAG_URL

# Weather Analysis Function
def weather_view(request):
    if request.method == 'POST':
        form = WeatherForm(request.POST)
        if form.is_valid():
            city = form.cleaned_data['city']
            # Normalize city name
            city = normalize_city_name(city)
        else:
            city = get_city_from_ip(request)
    else:
        form = WeatherForm()
        city = request.GET.get('city', '')
        
        if not city:
            # Try to get city from IP address
            city = get_city_from_ip(request)
        else:
            # Normalize city name from GET parameter
            city = normalize_city_name(city)
    
    # Handle empty city name by using default city
    if not city:
        city = normalize_city_name(DEFAULT_CITY)
    
    # Get current weather data for the specified city
    current_weather = get_current_weather(city)
    
    cod = current_weather.get('cod', 500)
    
    # Handle errors
    if cod != 200:
        context = {
            'cod': cod,
            'error_message': current_weather.get('message', 'Unknown error'),
            'city': city,
            'FLAG_URL': FLAG_URL,
            'form': form
        }
        return render(request, 'weather.html', context)

    # Get weather icon type based on description
    icon_type = get_weather_icon(current_weather['description'])

    # Find historical data file that matches the city or country code
    try:
        country_code = current_weather.get('country_code')
        historical_data_file = find_city_historical_data(
            HISTORICAL_DATA_PATH, 
            city=city, 
            country_code=country_code, 
            default_file="weather.csv"
        )
        
        print(historical_data_file)
        # Load historical data from the matched file
        historical_data = read_historical_data(historical_data_file)
        
        # If the historical data is empty, try to find alternative sources
        if historical_data.empty:
            # Try the default weather.csv file as fallback
            if historical_data_file != os.path.join(HISTORICAL_DATA_PATH, "weather.csv"):
                default_file_path = os.path.join(HISTORICAL_DATA_PATH, "weather.csv")
                if os.path.exists(default_file_path):
                    historical_data = read_historical_data(default_file_path)
        else:
            # Add city column if it doesn't exist
            if 'city' not in historical_data.columns:
                # Add city column to file data so it can be properly used
                historical_data['city'] = city
    except Exception as e:
        historical_data = pd.DataFrame()  # Use empty DataFrame as fallback
    
    # Load and enhance historical data
    enhanced_data = enrich_historical_data(historical_data, city)

    # Prepare data for temperature prediction only
    X_temp, y_temp = prepare_regression_data(enhanced_data)
    temp_model = train_regression_model(X_temp, y_temp)

    # Predict future daily temperature (5 days)
    forecast = predict_future(temp_model, current_weather['current_temp'], 
                            feature_name='Temp', 
                            past_window=3,
                            days=FORECAST_DAYS)

    # Get forecast dates
    forecast_dates = format_future_times(days=FORECAST_DAYS)
    
    # Create a list of forecast days with all information
    forecast_days = []
    for i in range(FORECAST_DAYS):
        day_forecast = {
            'date': forecast_dates[i],
            'min_temp': forecast['min_temps'][i],
            'max_temp': forecast['max_temps'][i],
            'description': forecast['descriptions'][i],
            'icon': get_weather_icon(forecast['descriptions'][i])
        }
        forecast_days.append(day_forecast)

    # Calculate temperature percentage for slider
    temp_percentage = calculate_temp_percentage(
        current_weather['temp_min'], 
        current_weather['temp_max'], 
        current_weather['current_temp']
    )

    # Prepare context for template
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
        'form': form
    }

    return render(request, 'weather.html', context)

# New view for historical capital city data
def capital_historical(request):
    country_code = request.GET.get('country', 'US').upper()
    # Get data source (NOAA or Open-Meteo, default to Open-Meteo)
    source = request.GET.get('source', 'open_meteo')
    
    try:
        # Set appropriate max days based on source
        max_days = 3650 if source.lower() == 'open_meteo' else 364
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
      