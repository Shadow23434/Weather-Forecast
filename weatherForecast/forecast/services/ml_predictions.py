import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error
import os
from .config import RANDOM_FOREST_PARAMS, TEST_SIZE, RANDOM_STATE, FORECAST_HOURS

def read_historical_data(filename):
    """Load and clean historical weather data"""
    df = pd.read_csv(filename)
    df = df.dropna()
    df = df.drop_duplicates()
    return df

def prepare_data(data):
    """Prepare data for training weather prediction models"""
    le = LabelEncoder()
    data['WindGustDir'] = le.fit_transform(data['WindGustDir'])
    data['RainTomorrow'] = le.fit_transform(data['RainTomorrow'])

    # Define feature variables and target variable
    X = data[['MinTemp', 'MaxTemp', 'WindGustDir', 'WindGustSpeed', 'Humidity', 'Pressure', 'Temp']]
    y = data['RainTomorrow']

    return X, y, le

def train_rain_model(X, y):
    """Train a model to predict rain occurrence"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE
    )
    
    model = RandomForestClassifier(
        n_estimators=RANDOM_FOREST_PARAMS['n_estimators'], 
        random_state=RANDOM_FOREST_PARAMS['random_state']
    )
    
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Mean Squared Error for Rain Model")
    print(mean_squared_error(y_test, y_pred))

    return model

def prepare_regression_data(data, feature, window_size=1):
    """
    Prepare data for regression models using a time series approach
    
    Args:
        data: DataFrame containing historical data
        feature: Feature name to predict
        window_size: Number of past values to use for prediction
    
    Returns:
        X: Input features - past values
        y: Target values - next value to predict
    """
    if window_size < 1:
        window_size = 1
    
    X, y = [], []
    
    if window_size == 1:
        # Original single-value approach
        for i in range(len(data) - 1):
            X.append(data[feature].iloc[i])
            y.append(data[feature].iloc[i+1])
        
        X = np.array(X).reshape(-1, 1)
    else:
        # Window-based approach
        for i in range(len(data) - window_size):
            # Get window of past values
            window = data[feature].iloc[i:i+window_size].values
            # Get next value to predict
            next_value = data[feature].iloc[i+window_size]
            
            X.append(window)
            y.append(next_value)
        
        X = np.array(X)
    
    y = np.array(y)
    return X, y

def train_regression_model(X, y):
    """
    Train a regression model for continuous value prediction
    
    Args:
        X: Input features - can be single value or window of values
        y: Target values
    
    Returns:
        Trained model
    """
    # Configure model with more trees for better accuracy
    model = RandomForestRegressor(
        n_estimators=RANDOM_FOREST_PARAMS['n_estimators'], 
        random_state=RANDOM_FOREST_PARAMS['random_state'],
        min_samples_leaf=2,  # Reduce overfitting
        max_features='sqrt'  # Better generalization
    )
    
    # Add early stopping if dataset is large enough
    if len(X) > 1000:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.1, random_state=RANDOM_FOREST_PARAMS['random_state']
        )
        model.fit(X_train, y_train)
        
        # Print validation score for monitoring
        val_score = model.score(X_val, y_val)
        print(f"Validation R² score: {val_score:.4f}")
    else:
        model.fit(X, y)
    
    return model

def predict_future(model, current_value, feature_name=None, historical_data=None, past_window=3):
    """
    Predict future values with enhanced accuracy using:
    1. Window-based prediction that considers multiple past values
    2. Error correction using historical trends
    3. Boundary checks to avoid unrealistic values
    
    Args:
        model: Trained regression model
        current_value: Latest observed value
        feature_name: Name of the feature being predicted (for boundary checks)
        historical_data: DataFrame with historical data (for trend analysis)
        past_window: Number of past values to consider for prediction
    
    Returns:
        List of predicted future values
    """
    predictions = [current_value]
    
    # Set realistic boundaries based on feature type
    if feature_name:
        if feature_name == 'Temp':
            min_valid, max_valid = -10, 50  # Reasonable temperature range in Celsius
        elif feature_name == 'Humidity':
            min_valid, max_valid = 0, 100   # Humidity percentage bounds
        elif feature_name == 'Pressure':
            min_valid, max_valid = 950, 1050  # Typical atmospheric pressure range (hPa)
        else:
            min_valid, max_valid = None, None
    else:
        min_valid, max_valid = None, None
    
    # Calculate average change from historical data if available
    avg_change = 0
    if historical_data is not None and feature_name is not None:
        if feature_name in historical_data.columns and len(historical_data) > 1:
            # Calculate average hour-to-hour change for this feature
            changes = historical_data[feature_name].diff().dropna()
            avg_change = changes.mean()
    
    # Prepare initial window of past values (start with current value repeated)
    window = [current_value] * past_window
    
    for i in range(FORECAST_HOURS):
        if past_window > 1:
            # Use multiple past values for prediction
            X_window = np.array(window[-past_window:]).reshape(1, -1)
            
            try:
                # Try to use window-based prediction if model supports it
                next_value = model.predict(X_window)[0]
            except:
                # Fallback to single-value prediction
                next_value = model.predict(np.array([[window[-1]]]))[0]
        else:
            # Use single value prediction
            next_value = model.predict(np.array([[window[-1]]]))[0]
        
        # Apply trend correction using historical average change
        if avg_change != 0:
            next_value += avg_change * 0.5  # Apply partial correction
        
        # Apply boundary checks if applicable
        if min_valid is not None and next_value < min_valid:
            next_value = min_valid
        if max_valid is not None and next_value > max_valid:
            next_value = max_valid
            
        # Store prediction and update window
        predictions.append(next_value)
        window.append(next_value)
    
    return predictions[1:]

def map_wind_direction(wind_deg, le):
    """Map wind direction degrees to compass points"""
    wind_deg = wind_deg % 360
    compass_points = [
        ("N", 0, 11.25), ("NNE", 11.25, 33.75), ("NE", 33.75, 56.25),
        ("ENE", 56.25, 78.75), ("E", 78.75, 101.25), ("ESE", 101.25, 123.75),
        ("SE", 123.75, 146.25), ("SSE", 146.25, 168.75), ("S", 168.75, 191.25),
        ("SSW", 191.25, 213.75), ("SW", 213.75, 236.25), ("WSW", 236.25, 258.75),
        ("W", 258.75, 281.25), ("WNW", 281.25, 303.75), ("NW", 303.75, 326.25),
        ("NNW", 326.25, 348.75)
    ]
    
    for direction, start, end in compass_points:
        if start <= wind_deg < end:
            compass_direction = direction
            break
    else:
        compass_direction = "N"  # Default case
        
    # Get encoded value if direction is in encoder classes
    return le.transform([compass_direction])[0] if compass_direction in le.classes_ else -1 