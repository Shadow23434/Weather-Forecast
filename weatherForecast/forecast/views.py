from django.shortcuts import render
from django.http import HttpResponse
from datetime import datetime
import os
import pandas as pd

# Import services
from .services import (
    get_current_weather, get_city_from_ip, get_weather_icon,
    read_historical_data, prepare_data, train_rain_model,
    prepare_regression_data, train_regression_model,
    predict_future, map_wind_direction,
    format_future_times, calculate_temp_percentage
)
from .services.config import DEFAULT_CITY, HISTORICAL_DATA_PATH, FORECAST_HOURS, FLAG_URL

# Weather Analysis Function
def weather_view(request):
    if request.method == 'POST':
        city = request.POST.get('city')
    else:
        city = get_city_from_ip(request)
        if not city:
            city = DEFAULT_CITY
    
    current_weather = get_current_weather(city)

    cod = current_weather['cod']
    if cod != 200:
        context = {
            'cod': cod,
            'error_message': current_weather['message'],
            'location': city,
        }
        return render(request, 'weather.html', context)

    # Get weather icon type based on description
    icon_type = get_weather_icon(current_weather['description'])

    # Load historical data
    historical_data = read_historical_data(HISTORICAL_DATA_PATH)

    # Prepare and train the rain prediction model
    X, y, le = prepare_data(historical_data)
    rain_model = train_rain_model(X, y)

    # Map wind direction to compass points and get encoded value
    compass_direction_encoded = map_wind_direction(current_weather['wind_gust_dir'], le)

    # Create current data for model input
    current_data = {
        'MinTemp': current_weather['temp_min'],
        'MaxTemp': current_weather['temp_max'],
        'WindGustDir': compass_direction_encoded,
        'WindGustSpeed': current_weather['wind_gust_speed'],
        'Humidity': current_weather['humidity'],
        'Pressure': current_weather['pressure'],
        'Temp': current_weather['current_temp']
    }
    current_df = pd.DataFrame([current_data])

    # Rain prediction
    rain_prediction = rain_model.predict(current_df)[0]

    # Prepare regression models for temperature and humidity
    X_temp, y_temp = prepare_regression_data(historical_data, 'Temp', window_size=3)
    X_hum, y_hum = prepare_regression_data(historical_data, 'Humidity', window_size=3)

    temp_model = train_regression_model(X_temp, y_temp)
    hum_model = train_regression_model(X_hum, y_hum)

    # Predict future temperature and humidity
    future_temp = predict_future(temp_model, current_weather['current_temp'], 
                              feature_name='Temp', 
                              historical_data=historical_data, 
                              past_window=3)
    future_humidity = predict_future(hum_model, current_weather['humidity'], 
                                  feature_name='Humidity', 
                                  historical_data=historical_data, 
                                  past_window=3)

    # Get future times
    future_times = format_future_times(hours=FORECAST_HOURS)
    
    # Store each value separately
    time1, time2, time3, time4, time5 = future_times
    temp1, temp2, temp3, temp4, temp5 = future_temp
    hum1, hum2, hum3, hum4, hum5 = future_humidity

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
        
        'time1': time1,
        'time2': time2,
        'time3': time3,
        'time4': time4,
        'time5': time5,

        'temp1': f"{round(temp1,1)}",
        'temp2': f"{round(temp2,1)}",
        'temp3': f"{round(temp3,1)}",
        'temp4': f"{round(temp4,1)}",
        'temp5': f"{round(temp5,1)}",

        'hum1': f"{round(hum1,1)}",
        'hum2': f"{round(hum2,1)}",
        'hum3': f"{round(hum3,1)}",
        'hum4': f"{round(hum4,1)}",
        'hum5': f"{round(hum5,1)}",
        
        'temp_percentage': temp_percentage,
        'FLAG_URL': FLAG_URL
    }

    return render(request, 'weather.html', context)
      