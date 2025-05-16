import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

import os
from datetime import datetime, timedelta
from .config import RANDOM_FOREST_PARAMS, TEST_SIZE, RANDOM_STATE, FORECAST_DAYS, DATA_DIRECTORY
from .weather_api import normalize_city_name


def read_historical_data(path):
    """
    Load and clean historical weather data.
    If the path is a directory, find the most recent CSV file in the directory.
    
    Args:
        path (str): Path to the CSV file or directory containing CSV files
    
    Returns:
        DataFrame: Cleaned weather data
    """
    try:
        # Check if path is a directory
        if os.path.isdir(path):
            # Find all CSV files in the directory
            csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
            
            if not csv_files:
                return pd.DataFrame()
            
            # Sort by modification time (newest first)
            csv_files.sort(key=lambda x: os.path.getmtime(os.path.join(path, x)), reverse=True)
            
            # Use the most recent CSV file
            file_path = os.path.join(path, csv_files[0])
        else:
            # This is a direct file path
            file_path = path
        
        # Read CSV file
        df = pd.read_csv(file_path)
        
        # Check if the DataFrame is empty
        if df.empty:
            return pd.DataFrame()
            
        # Check if we're dealing with weather data from Open-Meteo API
        open_meteo_columns = ['temp', 'temp_min', 'temp_max', 'humidity', 'pressure', 'wind_speed', 'wind_deg']
        is_open_meteo = any(col in df.columns for col in open_meteo_columns)
        
        # For Open-Meteo data, handle missing values accordingly
        if is_open_meteo:
            # Only drop rows where ALL these important columns are missing
            required_columns = [col for col in open_meteo_columns if col in df.columns]
            if required_columns:
                initial_len = len(df)
                df = df.dropna(subset=required_columns, how='all')
        else:
            # For other types of data, use more general cleaning
            df = df.dropna(how='all')  # Only drop rows that are completely empty
        
        # Always remove duplicate rows
        df = df.drop_duplicates()
        
        # Check if the cleaned DataFrame is empty
        if df.empty:
            return pd.DataFrame()
            
        return df
        
    except FileNotFoundError:
        return pd.DataFrame()
    except PermissionError:
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()

def find_city_historical_data(data_directory, city=None, country_code=None, default_file="weather.csv"):
    """
    Find a historical data file that matches the given city or country code.
    
    Args:
        data_directory (str): Directory containing historical data files
        city (str): City name to search for
        country_code (str): Country code to search for
        default_file (str): Default file to use if no matching file is found
        
    Returns:
        str: Path to the matched file or default file
    """
    if not os.path.isdir(data_directory):
        return os.path.join(data_directory, default_file)
    
    # Get all CSV files in the directory
    csv_files = [f for f in os.listdir(data_directory) if f.endswith('.csv')]
    
    if not csv_files:
        return os.path.join(data_directory, default_file)
    
    # Convert city and country_code to lowercase if provided
    if city:
        city = normalize_city_name(city)
    if country_code:
        country_code = country_code.lower().strip()
    
    # If city name is provided, search all files for the exact city match first
    if city:
        # First, prioritize files that might be specifically for this city (contain city name in filename)
        city_filename_matches = [f for f in csv_files if city in f.lower()]
        if city_filename_matches:
            # Check these files first
            for file in city_filename_matches:
                try:
                    file_path = os.path.join(data_directory, file)
                    df = pd.read_csv(file_path)
                    
                    if 'city' in df.columns:
                        # Convert all city names to lowercase for comparison
                        df_cities = df['city'].apply(normalize_city_name)
                        
                        # Check if the exact city name exists in the file
                        city_match = (df_cities == city).any()
                        
                        if city_match:
                            return file_path
                except Exception:
                    continue
        
        # Next, if country code is provided, check country-specific files
        if country_code:
            country_matches = [f for f in csv_files if country_code in f.lower()]
            for file in country_matches:
                try:
                    file_path = os.path.join(data_directory, file)
                    df = pd.read_csv(file_path)
                    
                    if 'city' in df.columns:
                        df_cities = df['city'].apply(normalize_city_name)
                        
                        # Check if the exact city name exists in the file
                        city_match = (df_cities == city).any()
                        
                        if city_match:
                            return file_path
                except Exception:
                    continue
        
        # Finally, check all remaining files for the city
        other_files = [f for f in csv_files if f not in (city_filename_matches + (country_matches if country_code else []))]
        for file in other_files:
            try:
                file_path = os.path.join(data_directory, file)
                df = pd.read_csv(file_path)
                
                if 'city' in df.columns:
                    df_cities = df['city'].apply(normalize_city_name)
                    
                    # Check if the exact city name exists in the file
                    city_match = (df_cities == city).any()
                    
                    if city_match:
                        return file_path
                else:
                    # For files without city column, check if the filename contains both city and country
                    if city in file.lower() and (not country_code or country_code in file.lower()):
                        return file_path
            except Exception:
                continue
    
    # If we're only looking for country-specific data (no city or city not found)
    if country_code:
        country_matches = [f for f in csv_files if country_code in f.lower()]
        if country_matches:
            # If city is also provided, prioritize files that contain the city name in the filename
            if city:
                city_country_matches = [f for f in country_matches if city in f.lower()]
                if city_country_matches:
                    # Sort by modification time (newest first)
                    city_country_matches.sort(key=lambda x: os.path.getmtime(os.path.join(data_directory, x)), reverse=True)
                    selected_file = city_country_matches[0]
                    return os.path.join(data_directory, selected_file)
            
            # Sort by modification time (newest first)
            country_matches.sort(key=lambda x: os.path.getmtime(os.path.join(data_directory, x)), reverse=True)
            
            return os.path.join(data_directory, country_matches[0])
    
    # If no match found for city or country, use the default file
    if default_file in csv_files:
        return os.path.join(data_directory, default_file)
    else:
        # If default file doesn't exist, use the most recent file
        csv_files.sort(key=lambda x: os.path.getmtime(os.path.join(data_directory, x)), reverse=True)
        return os.path.join(data_directory, csv_files[0])

def _deg_to_compass(deg):
    """Convert wind direction in degrees to compass direction"""
    directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    idx = round(deg / 22.5) % 16
    return directions[idx]

def enrich_historical_data(base_data, city=None):
    """
    Enrich historical weather data for machine learning, focusing only on temperature data
    
    Args:
        base_data (DataFrame): Raw historical weather data
        city (str): City to filter data by (optional)
    
    Returns:
        DataFrame: Enriched data ready for machine learning
    """
    # Copy to avoid modifying original
    df = base_data.copy()
    
    # If the DataFrame is empty, create a minimal synthetic dataset to ensure model training
    if df.empty or len(df) < 10:
        # Create a synthetic dataset with basic temperature patterns
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        synthetic_data = {
            'date': dates,
            'temp': [20 + 5 * np.sin(i/10) + np.random.normal(0, 1) for i in range(100)],
            'city': [city or 'Default'] * 100
        }
        df = pd.DataFrame(synthetic_data)
    
    # Convert city to lowercase if provided
    if city:
        city = normalize_city_name(city)
    
    # If city is provided and 'city' column exists, filter data by exact city name
    if city and 'city' in df.columns:
        # Convert all city names to lowercase for comparison
        df_cities = df['city'].apply(normalize_city_name)
        
        # Filter by exact city name match
        filtered_data = df[df_cities == city]
        
        # If we don't have data for this exact city, try a partial match as fallback
        if len(filtered_data) < 10:  # Need at least 10 records
            # Use lowercase for contains search as well
            filtered_data = df[df_cities.str.contains(city, na=False)]
            
            # If still not enough data, use all available data
            if len(filtered_data) < 10:
                filtered_data = df
    else:
        # If no city column exists, assume the entire file is for the requested city
        # (this handles city-specific files like JP_open_meteo_historical with Tokyo data)
        if city and 'city' not in df.columns:
            filtered_data = df
            
            # Add city column to the data for consistent processing downstream
            filtered_data['city'] = city
        else:
            # If no city column exists or no city specified, use all data
            filtered_data = df
    
    # Ensure datetime format
    if 'date' in filtered_data.columns and not pd.api.types.is_datetime64_dtype(filtered_data['date']):
        filtered_data['date'] = pd.to_datetime(filtered_data['date'])
    
    # Map standard column names
    column_mapping = {
        'temp': 'Temp',
        'temp_min': 'MinTemp',
        'temp_max': 'MaxTemp'
    }
    
    # Rename columns if they exist
    for old_col, new_col in column_mapping.items():
        if old_col in filtered_data.columns and new_col not in filtered_data.columns:
            filtered_data[new_col] = filtered_data[old_col]
    
    # Create default values for any missing required columns
    required_columns = ['Temp']
    for col in required_columns:
        if col not in filtered_data.columns:
            # Try to derive Temp from other temperature columns if available
            if 'temp_max' in filtered_data.columns and 'temp_min' in filtered_data.columns:
                filtered_data[col] = filtered_data[['temp_max', 'temp_min']].mean(axis=1)
            else:
                # If no temperature data available, use a default value
                filtered_data[col] = 20.0
    
    # Fill NaN values in Temp column with forward fill then backward fill
    if 'Temp' in filtered_data.columns:
        filtered_data['Temp'] = filtered_data['Temp'].ffill().bfill()
        
        # If there are still NaN values, fill with the mean or a default value
        if filtered_data['Temp'].isna().any():
            mean_temp = filtered_data['Temp'].mean()
            if np.isnan(mean_temp):
                mean_temp = 20.0  # Default value if mean is NaN
            filtered_data['Temp'] = filtered_data['Temp'].fillna(mean_temp)
    
    return filtered_data

def prepare_data(data):
    """
    Prepare data for rain prediction model
    
    Args:
        data (DataFrame): Weather data
        
    Returns:
        tuple: X (features), y (target), label_encoder
    """
    # Select relevant features
    features = ['MinTemp', 'MaxTemp', 'WindGustDir', 'WindGustSpeed', 'Humidity', 'Pressure', 'Temp']
    
    # Encode categorical variables
    le = LabelEncoder()
    data['WindGustDir_encoded'] = le.fit_transform(data['WindGustDir'])
    
    # Form feature matrix
    X = data[['MinTemp', 'MaxTemp', 'WindGustDir_encoded', 'WindGustSpeed', 'Humidity', 'Pressure', 'Temp']]
    
    # Convert rain categories to binary
    y = (data['RainTomorrow'] == 'True').astype(int)
    
    return X, y, le

def train_rain_model(X, y):
    """
    Train random forest model for rain prediction
    
    Args:
        X (DataFrame): Feature matrix
        y (Series): Target variable
        
    Returns:
        RandomForestClassifier: Trained model
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    # Create and train model
    model = RandomForestClassifier(**RANDOM_FOREST_PARAMS)
    model.fit(X_train, y_train)
    
    # Evaluate model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Rain Model - Train accuracy: {train_score:.3f}, Test accuracy: {test_score:.3f}")
    
    return model

def prepare_regression_data(data, feature='Temp', window_size=3):
    """
    Prepare data for temperature regression model with time-lag features
    
    Args:
        data (DataFrame): Weather data
        feature (str): Target feature to predict (default: 'Temp')
        window_size (int): Number of previous time steps to use
    
    Returns:
        tuple: X (features), y (target)
    """
    # Check if data is empty or None
    if data is None or data.empty:
        # Create synthetic data for minimal model training
        synthetic_data = pd.DataFrame({
            feature: [20 + i*0.1 + np.random.normal(0, 0.5) for i in range(30)]
        })
        data = synthetic_data
    
    # Ensure feature exists in the data
    if feature not in data.columns:
        # Try to find similar column names
        temp_columns = [col for col in data.columns if 'temp' in col.lower()]
        if temp_columns:
            # Use the first available temperature column
            data[feature] = data[temp_columns[0]]
        else:
            # Create synthetic data if no temperature columns exist
            data[feature] = [20 + i*0.1 + np.random.normal(0, 0.5) for i in range(len(data))]
    
    # Filter out rows with NaN in target column
    filtered_data = data[data[feature].notna()].copy()
    
    # Ensure we have enough data
    if len(filtered_data) < window_size + 1:
        # Add synthetic rows to meet the minimum requirement
        synthetic_rows = window_size + 10 - len(filtered_data)
        if synthetic_rows > 0:
            last_value = filtered_data[feature].iloc[-1] if not filtered_data.empty else 20.0
            synthetic_data = pd.DataFrame({
                feature: [last_value + i*0.1 + np.random.normal(0, 0.5) for i in range(synthetic_rows)]
            })
            filtered_data = pd.concat([filtered_data, synthetic_data])
    
    # Create lag features
    for i in range(1, window_size + 1):
        filtered_data[f'{feature}_lag_{i}'] = filtered_data[feature].shift(i)
    
    # Drop rows with NaN (first window_size rows)
    filtered_data = filtered_data.dropna()
    
    # Double-check that we have data after creating lag features
    if filtered_data.empty:
        # Create minimal synthetic data for model training
        synthetic_rows = window_size + 10
        base_value = 20.0
        synthetic_data = pd.DataFrame({
            feature: [base_value + i*0.1 + np.random.normal(0, 0.5) for i in range(synthetic_rows)]
        })
        # Create lag features for synthetic data
        for i in range(1, window_size + 1):
            synthetic_data[f'{feature}_lag_{i}'] = synthetic_data[feature].shift(i)
        # Drop rows with NaN values
        synthetic_data = synthetic_data.dropna()
        filtered_data = synthetic_data
    
    # Create X and y
    X = filtered_data[[f'{feature}_lag_{i}' for i in range(1, window_size + 1)]]
    y = filtered_data[feature]
    
    return X, y

def train_lstm_model(X, y, epochs=30, batch_size=16):
    """
    Huấn luyện mô hình LSTM cho dự báo thời tiết.
    Args:
        X (DataFrame): Dữ liệu đầu vào (n_samples, n_features)
        y (Series): Nhãn đầu ra
    Returns:
        model: Mô hình LSTM đã huấn luyện
    """
    import numpy as np
    # Đảm bảo X là numpy array
    X_np = np.array(X)
    y_np = np.array(y)
    # LSTM yêu cầu input shape: (samples, timesteps, features)
    # Ở đây, mỗi sample là 1 chuỗi các giá trị lag, coi như 1 timestep
    X_lstm = X_np.reshape((X_np.shape[0], X_np.shape[1], 1))
    model = Sequential()
    model.add(LSTM(32, input_shape=(X_lstm.shape[1], 1), return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_lstm, y_np, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[es], verbose=0)
    return model

def train_regression_model(X, y):
    # Loại bỏ các feature toàn NaN hoặc hằng số
    X = X.loc[:, X.nunique() > 1]
    X = X.dropna(axis=1, how='all')
    if X.empty or y is None or y.empty:
        X_default = pd.DataFrame({f'Temp_lag_{i}': [20 + i + j*0.1 for j in range(10)] for i in range(1, 4)})
        y_default = pd.Series([20 + i*0.2 for i in range(10)])
        scaler = StandardScaler().fit(X_default)
        X_default_scaled = scaler.transform(X_default)
        rf_model = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=RANDOM_STATE)
        rf_model.fit(X_default_scaled, y_default)
        lstm_model = train_lstm_model(X_default_scaled, y_default)
        lgbm_model = None
        if lgb is not None:
            lgbm_model = lgb.LGBMRegressor(n_estimators=10, max_depth=3, min_data_in_leaf=1, min_data_in_bin=1, random_state=RANDOM_STATE)
            lgbm_model.fit(X_default_scaled, y_default)
        return rf_model, None, None, None, None, None, lstm_model, lgbm_model

    if len(X) < 10:
        n_needed = 10 - len(X)
        X_synthetic = pd.DataFrame({col: [X[col].mean() + np.random.normal(0, 1) for _ in range(n_needed)] for col in X.columns})
        y_synthetic = pd.Series([y.mean() + np.random.normal(0, 1) for _ in range(n_needed)])
        X = pd.concat([X, X_synthetic], ignore_index=True)
        y = pd.concat([y, y_synthetic], ignore_index=True)

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
    except ValueError:
        X_train, y_train = X_scaled, y
        X_test, y_test = X_scaled[:1], y.iloc[:1]

    rf_params = {'n_estimators': 50, 'max_depth': 10, 'min_samples_split': 2, 'min_samples_leaf': 1, 'random_state': RANDOM_STATE}
    rf_model = RandomForestRegressor(**rf_params)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    print(f"RandomForest RMSE: {np.sqrt(mean_squared_error(y_test, rf_pred)):.3f}, MAE: {mean_absolute_error(y_test, rf_pred):.3f}, R2: {r2_score(y_test, rf_pred):.3f}")

    xgb_params = {'n_estimators': 50, 'max_depth': 6, 'learning_rate': 0.1, 'random_state': RANDOM_STATE}
    xgb_model = XGBRegressor(**xgb_params)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    print(f"XGBoost RMSE: {np.sqrt(mean_squared_error(y_test, xgb_pred)):.3f}, MAE: {mean_absolute_error(y_test, xgb_pred):.3f}, R2: {r2_score(y_test, xgb_pred):.3f}")

    lgbm_model = None
    lgbm_pred = np.zeros_like(rf_pred)
    if lgb is not None and X_train.shape[1] > 0:
        lgbm_model = lgb.LGBMRegressor(n_estimators=50, max_depth=10, min_data_in_leaf=1, min_data_in_bin=1, random_state=RANDOM_STATE)
        lgbm_model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
        lgbm_pred = lgbm_model.predict(X_test)
        print(f"LightGBM RMSE: {np.sqrt(mean_squared_error(y_test, lgbm_pred)):.3f}, MAE: {mean_absolute_error(y_test, lgbm_pred):.3f}, R2: {r2_score(y_test, lgbm_pred):.3f}")

    lstm_model = train_lstm_model(X_train, y_train)
    X_test_lstm = np.array(X_test).reshape((X_test.shape[0], X_test.shape[1], 1))
    lstm_pred = lstm_model.predict(X_test_lstm, verbose=0).flatten()
    print(f"LSTM RMSE: {np.sqrt(mean_squared_error(y_test, lstm_pred)):.3f}, MAE: {mean_absolute_error(y_test, lstm_pred):.3f}, R2: {r2_score(y_test, lstm_pred):.3f}")

    # Kết hợp stacking với 4 mô hình
    meta_X = np.column_stack((rf_pred, xgb_pred, lstm_pred, lgbm_pred))
    meta_X_train, meta_X_test, meta_y_train, meta_y_test = train_test_split(
        meta_X, y_test, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    meta_model = LinearRegression()
    meta_model.fit(meta_X_train, meta_y_train)
    final_pred = meta_model.predict(meta_X_test)
    rmse = np.sqrt(mean_squared_error(meta_y_test, final_pred)) if len(meta_y_test) >= 2 else None
    mae = mean_absolute_error(meta_y_test, final_pred) if len(meta_y_test) >= 2 else None
    r2 = r2_score(meta_y_test, final_pred) if len(meta_y_test) >= 2 else None
    print(f"Stacking RMSE: {rmse:.3f}, MAE: {mae:.3f}, R2: {r2:.3f}")
    return rf_model, xgb_model, meta_model, rmse, mae, r2, lstm_model, lgbm_model

def predict_future_stacking(rf_model, xgb_model, meta_model, current_value, feature_name='Temp', past_window=3, days=5, lstm_model=None, lgbm_model=None):
    """
    Dự báo tương lai sử dụng stacking, có thêm LSTM và LightGBM.
    """
    import numpy as np
    min_temps = []
    max_temps = []
    descriptions = []
    weather_conditions = {
        "hot": ["Clear sky", "Sunny", "Partly cloudy", "Hot"],
        "warm": ["Clear sky", "Partly cloudy", "Cloudy", "Light rain"],
        "mild": ["Partly cloudy", "Cloudy", "Light rain", "Mild"],
        "cool": ["Cloudy", "Light rain", "Rain", "Cool"],
        "cold": ["Light snow", "Snow", "Freezing", "Cold"]
    }
    recent_values = [float(current_value)] * past_window
    daily_fluctuation = 5.0
    for day in range(days):
        input_arr = np.array(recent_values).reshape((1, past_window))
        input_df = pd.DataFrame({f"{feature_name}_lag_{i+1}": [recent_values[i]] for i in range(past_window)})
        rf_pred = rf_model.predict(input_df)[0] if rf_model is not None else 0
        xgb_pred = xgb_model.predict(input_df)[0] if xgb_model is not None else 0
        lgbm_pred = lgbm_model.predict(input_df)[0] if lgbm_model is not None else 0
        if lstm_model is not None:
            input_lstm = input_arr.reshape((1, past_window, 1))
            lstm_pred = lstm_model.predict(input_lstm, verbose=0)[0][0]
        else:
            lstm_pred = 0
        # Kết hợp stacking nếu meta_model có, nếu không thì trung bình các mô hình
        preds = [p for p in [rf_pred, xgb_pred, lstm_pred, lgbm_pred] if p != 0]
        if meta_model is not None and len(preds) > 0:
            meta_input = np.array([[rf_pred, xgb_pred, lstm_pred, lgbm_pred]])
            base_temp = meta_model.predict(meta_input)[0]
        elif len(preds) > 0:
            base_temp = np.mean(preds)
        else:
            base_temp = float(current_value)
        base_temp = max(-40, min(base_temp, 50))
        seasonal_variation = 0.5 * np.sin(day/5)
        base_temp += seasonal_variation
        random_variation = np.random.normal(0, 0.7)
        min_temp = base_temp - (daily_fluctuation / 2) + random_variation
        max_temp = base_temp + (daily_fluctuation / 2) + random_variation
        if min_temp >= max_temp:
            mid_temp = (min_temp + max_temp) / 2
            min_temp = mid_temp - 2
            max_temp = mid_temp + 2
        if max_temp > 30:
            category = "hot"
        elif max_temp > 25:
            category = "warm"
        elif max_temp > 15:
            category = "mild"
        elif max_temp > 5:
            category = "cool"
        else:
            category = "cold"
        descriptions_for_category = weather_conditions[category]
        description_index = (day + np.random.randint(0, 2)) % len(descriptions_for_category)
        description = descriptions_for_category[description_index]
        min_temps.append(round(min_temp, 1))
        max_temps.append(round(max_temp, 1))
        descriptions.append(description)
        recent_values = [base_temp] + recent_values[:-1]
    return {
        'min_temps': min_temps,
        'max_temps': max_temps,
        'descriptions': descriptions
    }

def map_wind_direction(wind_deg, le):
    """
    Map wind direction in degrees to encoded value
    
    Args:
        wind_deg: Wind direction in degrees
        le: Label encoder fitted on wind directions
        
    Returns:
        int: Encoded wind direction
    """
    compass_dir = _deg_to_compass(wind_deg)
    try:
        return le.transform([compass_dir])[0]
    except ValueError:
        # If direction not in encoder, return most common
        return le.transform([le.classes_[0]])[0]
