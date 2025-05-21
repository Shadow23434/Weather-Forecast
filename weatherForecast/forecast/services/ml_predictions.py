import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import lightgbm as lgb
import optuna
import logging
import os
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import warnings
import joblib
from datetime import datetime, timedelta
from .config import RANDOM_STATE
from .weather_api import normalize_city_name


warnings.filterwarnings('ignore')
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add model storage path
MODEL_STORAGE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
os.makedirs(MODEL_STORAGE_PATH, exist_ok=True)

def get_model_path(city, target_type):
    """Get the path for a specific model"""
    city = normalize_city_name(city)
    return os.path.join(MODEL_STORAGE_PATH, f"{city}_{target_type}_model.joblib")

def save_model(model, scaler, features, city, target_type):
    """Save model and its components"""
    model_data = {
        'model': model,
        'scaler': scaler,
        'features': features,
        'last_trained': datetime.now()
    }
    model_path = get_model_path(city, target_type)
    joblib.dump(model_data, model_path)
    logger.info(f"Model saved to {model_path}")

def load_model(city, target_type):
    """Load model and its components"""
    model_path = get_model_path(city, target_type)
    if os.path.exists(model_path):
        model_data = joblib.load(model_path)
        # Check if model is older than 7 days
        if datetime.now() - model_data['last_trained'] > timedelta(days=7):
            logger.info(f"Model for {city} is older than 7 days, will retrain")
            return None
        return model_data
    return None

def train_and_save_model(csv_path, city, target_type):
    """Train model and save it for future use"""
    model, scaler, features, _ = forecast_temperature_from_csv(
        csv_path, 
        target=target_type,
        n_splits=3,
        n_estimators=100,
        random_state=RANDOM_STATE
    )
    if model is not None:
        save_model(model, scaler, features, city, target_type)
    return model, scaler, features

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

def optimize_hyperparameters(X, y, model_type='rf'):
    """
    Optimize hyperparameters with reduced search space
    """
    def objective(trial):
        if model_type == 'rf':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 8),
                'n_jobs': -1
            }
            model = RandomForestRegressor(**params, random_state=RANDOM_STATE)
        elif model_type == 'xgb':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'n_jobs': -1,
                'tree_method': 'hist'
            }
            model = XGBRegressor(**params, random_state=RANDOM_STATE)
        
        tscv = TimeSeriesSplit(n_splits=3)  # Reduced number of splits
        scores = []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = -mean_squared_error(y_test, y_pred)
            scores.append(score)
        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30, n_jobs=-1)  # Reduced number of trials
    return study.best_params

def create_advanced_features(df):
    """
    Create advanced features for the model with optimized processing
    """
    # Time-based features (reduced set)
    df['hour'] = df['date'].dt.hour
    df['day_of_year'] = df['date'].dt.dayofyear
    
    # Cyclical features (reduced set)
    df['sin_day'] = np.sin(2 * np.pi * df['day_of_year']/365)
    df['cos_day'] = np.cos(2 * np.pi * df['day_of_year']/365)
    
    # Reduced lag features (only most important lags)
    lag_columns = ['temp', 'temp_min', 'temp_max']
    for col in lag_columns:
        if col in df.columns:
            df[f'{col}_lag_1'] = df[col].shift(1)
            df[f'{col}_lag_7'] = df[col].shift(7)
    
    # Simplified rolling statistics
    for col in ['temp', 'temp_min', 'temp_max']:
        if col in df.columns:
            df[f'{col}_rolling_mean_7'] = df[col].rolling(window=7, min_periods=1).mean()
    
    # Essential interaction features only
    if all(col in df.columns for col in ['temp', 'humidity']):
        df['temp_humidity'] = df['temp'] * df['humidity']
    
    return df

def create_ensemble_model():
    """
    Create a simplified ensemble model
    """
    # Reduced number of base models
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    
    xgb = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        n_jobs=-1,
        tree_method='hist',
        random_state=RANDOM_STATE
    )
    
    # Simplified meta model
    meta_model = RandomForestRegressor(
        n_estimators=50,
        max_depth=5,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    
    # Create stacking ensemble with fewer models
    estimators = [
        ('rf', rf),
        ('xgb', xgb)
    ]
    
    ensemble = StackingRegressor(
        estimators=estimators,
        final_estimator=meta_model,
        cv=3,  # Reduced cross-validation folds
        n_jobs=-1
    )
    
    return ensemble

def forecast_temperature_from_csv(
    csv_path, 
    target,
    n_splits=3,
    n_estimators=100, 
    random_state=42,
    forecast_days=5,
    city=None  # Add city parameter
):
    # Try to load pre-trained model if city is provided
    if city:
        model_data = load_model(city, target)
        if model_data is not None:
            logger.info(f"Using pre-trained model for {city}")
            model = model_data['model']
            scaler = model_data['scaler']
            features = model_data['features']
            
            # Read data for prediction
            df = pd.read_csv(csv_path)
            df['date'] = pd.to_datetime(df['date'])
            
            # Create features for prediction
            df = create_advanced_features(df)
            
            # Select features
            features = [f for f in features if f in df.columns]
            
            # Handle missing values
            df[features] = df[features].fillna(method='ffill')
            df = df.dropna(subset=features)
            
            if len(df) == 0:
                logger.warning("Not enough data for prediction")
                return None, None, features, [None]*forecast_days
            
            X = df[features].values
            X_scaled = scaler.transform(X)
            
            # Make predictions
            last_row = df.iloc[-1].copy()
            forecast_results = []
            
            # Calculate historical statistics
            temp_mean = df[target].mean()
            temp_std = df[target].std()
            temp_min = df[target].min()
            temp_max = df[target].max()
            
            # Generate base forecast with trend
            base_trend = np.linspace(0, 2, forecast_days)
            
            for i in range(forecast_days):
                # Update time features
                next_date = last_row['date'] + pd.Timedelta(days=1)
                last_row['date'] = next_date
                last_row['hour'] = next_date.hour
                last_row['day_of_year'] = next_date.dayofyear
                last_row['sin_day'] = np.sin(2 * np.pi * next_date.dayofyear/365)
                last_row['cos_day'] = np.cos(2 * np.pi * next_date.dayofyear/365)
                
                # Prepare input features
                input_features = []
                for f in features:
                    input_features.append(last_row[f])
                input_scaled = scaler.transform([input_features])
                
                # Get prediction
                y_pred = model.predict(input_scaled)[0]
                
                # Add seasonal variation
                seasonal_factor = np.sin(2 * np.pi * next_date.dayofyear/365) * (temp_std * 0.5)
                
                # Add random variation
                if target == 'temp_min':
                    noise = np.random.normal(-1.0, 0.5 + i*0.2)
                elif target == 'temp_max':
                    noise = np.random.normal(1.0, 0.5 + i*0.2)
                else:
                    noise = np.random.normal(0, 0.3 + i*0.1)
                
                # Combine all factors
                y_pred = y_pred + seasonal_factor + noise + base_trend[i]
                
                # Ensure reasonable temperature range
                if target == 'temp_min':
                    y_pred = max(min(y_pred, temp_mean - 2), temp_min)
                elif target == 'temp_max':
                    y_pred = min(max(y_pred, temp_mean + 2), temp_max)
                else:
                    y_pred = max(min(y_pred, temp_max), temp_min)
                
                forecast_results.append(round(y_pred, 1))
                last_row[target] = y_pred
            
            return model, scaler, features, forecast_results
    
    # If no pre-trained model or city not provided, train new model
    # Read data
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Create optimized features
    df = create_advanced_features(df)
    
    # Select essential features only
    features = [
        'temp', 'temp_min', 'temp_max',
        'day_of_year', 'sin_day', 'cos_day'
    ]
    
    # Add essential lag features
    for col in ['temp', 'temp_min', 'temp_max']:
        if col in df.columns:
            features.extend([f'{col}_lag_1', f'{col}_lag_7'])
    
    # Add essential rolling features
    for col in ['temp', 'temp_min', 'temp_max']:
        if col in df.columns:
            features.extend([f'{col}_rolling_mean_7'])
    
    # Add essential interaction features
    if all(col in df.columns for col in ['temp', 'humidity']):
        features.extend(['temp_humidity'])
    
    features = [f for f in features if f in df.columns]
    
    # Handle missing values efficiently
    df[features] = df[features].fillna(method='ffill')
    df[target] = df[target].fillna(method='ffill')
    df = df.dropna(subset=features + [target])
    
    if len(df) == 0:
        print("Not enough data to train the model.")
        return None, None, features, [None]*forecast_days
    
    X = df[features].values
    y = df[target].values
    
    # Use StandardScaler instead of RobustScaler for faster processing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Optimize hyperparameters with reduced search space
    best_params = optimize_hyperparameters(X_scaled, y, model_type='rf')
    
    # Create and train simplified ensemble model
    ensemble = create_ensemble_model()
    ensemble.fit(X_scaled, y)
    
    # Forecast for the next 5 days
    last_row = df.iloc[-1].copy()
    forecast_results = []
    
    # Calculate historical statistics
    temp_mean = df[target].mean()
    temp_std = df[target].std()
    temp_min = df[target].min()
    temp_max = df[target].max()
    
    # Generate base forecast with trend
    base_trend = np.linspace(0, 2, forecast_days)
    
    for i in range(forecast_days):
        # Update time features
        next_date = last_row['date'] + pd.Timedelta(days=1)
        last_row['date'] = next_date
        last_row['hour'] = next_date.hour
        last_row['day_of_year'] = next_date.dayofyear
        last_row['sin_day'] = np.sin(2 * np.pi * next_date.dayofyear/365)
        last_row['cos_day'] = np.cos(2 * np.pi * next_date.dayofyear/365)
        
        # Prepare input features
        input_features = []
        for f in features:
            input_features.append(last_row[f])
        input_scaled = scaler.transform([input_features])
        
        # Get prediction from ensemble
        y_pred = ensemble.predict(input_scaled)[0]
        
        # Add seasonal variation
        seasonal_factor = np.sin(2 * np.pi * next_date.dayofyear/365) * (temp_std * 0.5)
        
        # Add random variation with increasing uncertainty
        if target == 'temp_min':
            noise = np.random.normal(-1.0, 0.5 + i*0.2)
        elif target == 'temp_max':
            noise = np.random.normal(1.0, 0.5 + i*0.2)
        else:
            noise = np.random.normal(0, 0.3 + i*0.1)
        
        # Combine all factors
        y_pred = y_pred + seasonal_factor + noise + base_trend[i]
        
        # Ensure reasonable temperature range
        if target == 'temp_min':
            y_pred = max(min(y_pred, temp_mean - 2), temp_min)
        elif target == 'temp_max':
            y_pred = min(max(y_pred, temp_mean + 2), temp_max)
        else:
            y_pred = max(min(y_pred, temp_max), temp_min)
        
        forecast_results.append(round(y_pred, 1))
        
        # Update time features
        last_row['date'] = next_date
        last_row['hour'] = next_date.hour
        last_row['day_of_year'] = next_date.dayofyear
        last_row['sin_day'] = np.sin(2 * np.pi * next_date.dayofyear/365)
        last_row['cos_day'] = np.cos(2 * np.pi * next_date.dayofyear/365)
        
        last_row[target] = y_pred
    
    return ensemble, scaler, features, forecast_results
