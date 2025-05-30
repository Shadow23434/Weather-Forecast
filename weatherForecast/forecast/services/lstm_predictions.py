import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import joblib
import json
from contextlib import contextmanager
matplotlib.use('Agg')  # Use Agg backend to avoid Tkinter issues

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model storage path
MODEL_STORAGE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
os.makedirs(MODEL_STORAGE_PATH, exist_ok=True)

@contextmanager
def safe_plot_context():
    """Context manager to safely handle matplotlib plots"""
    try:
        yield
    finally:
        plt.close('all')

def get_model_path(city, target_type):
    """Get the path for a specific model"""
    return os.path.join(MODEL_STORAGE_PATH, f"{city}_{target_type}_lstm_model.h5")

def get_scaler_path(city, target_type):
    """Get the path for a specific scaler"""
    return os.path.join(MODEL_STORAGE_PATH, f"{city}_{target_type}_lstm_scaler.joblib")

def calculate_metrics(y_true, y_pred):
    """Calculate various evaluation metrics"""
    try:
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {
            'MAE': round(mae, 2),
            'MSE': round(mse, 2),
            'RMSE': round(rmse, 2),
            'MAPE': round(mape, 2)
        }
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        return {
            'MAE': None,
            'MSE': None,
            'RMSE': None,
            'MAPE': None
        }

def plot_loss_history(history, city, target_type):
    """Plot training and validation loss history"""
    with safe_plot_context():
        plt.figure(figsize=(12, 6))
        
        # Plot training & validation loss
        plt.plot(history.history['loss'], 'b-', label='Training Loss', linewidth=2)
        plt.plot(history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        plt.title(f'Model Loss History - {city} {target_type}', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True)
        
        # Save plot
        plot_path = os.path.join(MODEL_STORAGE_PATH, f"{city}_{target_type}_loss_history.png")
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()

def plot_accuracy_history(history, city, target_type):
    """Plot training and validation accuracy history"""
    with safe_plot_context():
        plt.figure(figsize=(12, 6))
        
        try:
            # Get MAE values
            training_mae = np.array(history.history['mae'])
            val_mae = np.array(history.history['val_mae'])
            
            # Log original MAE values for verification
            logger.info(f"Original MAE values for {city} {target_type}:")
            logger.info(f"Training MAE - Min: {np.min(training_mae):.4f}, Max: {np.max(training_mae):.4f}, Mean: {np.mean(training_mae):.4f}")
            logger.info(f"Validation MAE - Min: {np.min(val_mae):.4f}, Max: {np.max(val_mae):.4f}, Mean: {np.mean(val_mae):.4f}")
            
            # Calculate accuracy using a more robust method
            # Normalize MAE to [0,1] range and convert to accuracy percentage
            max_mae = max(np.max(training_mae), np.max(val_mae))
            training_accuracy = 100 * (1 - training_mae / max_mae)
            val_accuracy = 100 * (1 - val_mae / max_mae)
            
            # Log calculated accuracy values for verification
            logger.info(f"Calculated accuracy values for {city} {target_type}:")
            logger.info(f"Training Accuracy - Min: {np.min(training_accuracy):.2f}%, Max: {np.max(training_accuracy):.2f}%, Mean: {np.mean(training_accuracy):.2f}%")
            logger.info(f"Validation Accuracy - Min: {np.min(val_accuracy):.2f}%, Max: {np.max(val_accuracy):.2f}%, Mean: {np.mean(val_accuracy):.2f}%")
            
            # Plot training & validation accuracy
            plt.plot(training_accuracy, 'g-', label='Training Accuracy', linewidth=2)
            plt.plot(val_accuracy, 'm-', label='Validation Accuracy', linewidth=2)
            
            # Add mean accuracy lines
            plt.axhline(y=np.mean(training_accuracy), color='g', linestyle='--', alpha=0.3, label='Mean Training Accuracy')
            plt.axhline(y=np.mean(val_accuracy), color='m', linestyle='--', alpha=0.3, label='Mean Validation Accuracy')
            
            plt.title(f'Model Accuracy History - {city} {target_type}', fontsize=14)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Accuracy (%)', fontsize=12)
            plt.legend(fontsize=12)
            plt.grid(True)
            
            # Set y-axis limits to ensure proper display
            plt.ylim(0, 100)
            
            # Save plot
            plot_path = os.path.join(MODEL_STORAGE_PATH, f"{city}_{target_type}_accuracy_history.png")
            plt.savefig(plot_path, bbox_inches='tight', dpi=300)
            plt.close()
            
            return True
        except Exception as e:
            logger.error(f"Error plotting accuracy history for {city} {target_type}: {str(e)}")
            return False

def plot_training_history(history, city, target_type):
    """Plot training history (both loss and accuracy)"""
    # Plot loss history
    plot_loss_history(history, city, target_type)
    
    # Plot accuracy history
    plot_accuracy_history(history, city, target_type)

def plot_predictions(y_true, y_pred, city, target_type):
    """Plot actual vs predicted values"""
    with safe_plot_context():
        plt.figure(figsize=(12, 6))
        plt.plot(y_true, label='Actual', marker='o')
        plt.plot(y_pred, label='Predicted', marker='x')
        plt.title(f'Actual vs Predicted - {city} {target_type}')
        plt.xlabel('Time')
        plt.ylabel('Temperature')
        plt.legend()
        
        # Save plot
        plot_path = os.path.join(MODEL_STORAGE_PATH, f"{city}_{target_type}_predictions.png")
        plt.savefig(plot_path)
        plt.close()

def build_lstm_model(seq_length, n_features):
    """Build LSTM model architecture"""
    try:
        model = Sequential([
            LSTM(128, input_shape=(seq_length, n_features), return_sequences=True),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['mae']  # Add Mean Absolute Error as a metric
        )
        return model
    except Exception as e:
        logger.error(f"Error building LSTM model: {str(e)}")
        return None

def prepare_data(df, target, seq_length=7):
    """Prepare data for LSTM training"""
    try:
        # Select features
        features = ['temp', 'temp_min', 'temp_max', 'humidity', 'pressure', 'wind_speed']
        features = [f for f in features if f in df.columns]
        
        # Create time-based features
        df['hour'] = df['date'].dt.hour
        df['day_of_year'] = df['date'].dt.dayofyear
        df['sin_day'] = np.sin(2 * np.pi * df['day_of_year']/365)
        df['cos_day'] = np.cos(2 * np.pi * df['day_of_year']/365)
        
        # Add features
        features.extend(['hour', 'sin_day', 'cos_day'])
        
        # Handle missing values
        df[features] = df[features].fillna(method='ffill')
        df[target] = df[target].fillna(method='ffill')
        df = df.dropna(subset=features + [target])
        
        if len(df) < seq_length + 1:
            raise ValueError(f"Not enough data after cleaning. Need at least {seq_length + 1} samples.")
        
        # Create separate scalers for features and target
        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()
        
        # Scale features and target separately
        scaled_features = feature_scaler.fit_transform(df[features])
        scaled_target = target_scaler.fit_transform(df[[target]])
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_features) - seq_length):
            X.append(scaled_features[i:(i + seq_length)])
            y.append(scaled_target[i + seq_length])
        
        X = np.array(X)
        y = np.array(y).reshape(-1)
        
        # Split into train, validation and test sets (70-15-15 split)
        train_size = int(len(X) * 0.7)
        val_size = int(len(X) * 0.15)
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:train_size+val_size]
        y_val = y[train_size:train_size+val_size]
        X_test = X[train_size+val_size:]
        y_test = y[train_size+val_size:]
        
        return X_train, y_train, X_val, y_val, X_test, y_test, (feature_scaler, target_scaler), features
    except Exception as e:
        logger.error(f"Error preparing data: {str(e)}")
        return None, None, None, None, None, None, None, None

def train_lstm_model(csv_path, city, target_type, seq_length=7):
    """Train LSTM model and save it"""
    try:
        # Read and prepare data
        df = pd.read_csv(csv_path)
        if len(df) < seq_length + 1:
            logger.error(f"Not enough data for training in {csv_path}")
            return None, None, None
            
        df['date'] = pd.to_datetime(df['date'])
        
        X_train, y_train, X_val, y_val, X_test, y_test, scalers, features = prepare_data(
            df, target_type, seq_length
        )
        
        if X_train is None:
            return None, None, None
            
        feature_scaler, target_scaler = scalers
        
        # Build and train model
        model = build_lstm_model(seq_length, len(features))
        if model is None:
            return None, None, None
        
        # Callbacks
        model_checkpoint = ModelCheckpoint(
            get_model_path(city, target_type),
            monitor='val_loss',
            save_best_only=True
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=[model_checkpoint],
            verbose=1
        )
        
        # Plot training history
        plot_training_history(history, city, target_type)
        
        # Make predictions on test set
        y_pred = model.predict(X_test)
        
        # Inverse transform predictions and actual values
        y_pred = target_scaler.inverse_transform(y_pred.reshape(-1, 1))
        y_test_orig = target_scaler.inverse_transform(y_test.reshape(-1, 1))
        
        # Calculate metrics
        metrics = calculate_metrics(y_test_orig, y_pred)
        logger.info(f"Test metrics for {city} {target_type}: {metrics}")
        
        # Plot predictions
        plot_predictions(y_test_orig, y_pred, city, target_type)
        
        # Save metrics
        metrics_path = os.path.join(MODEL_STORAGE_PATH, f"{city}_{target_type}_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f)
        
        # Save scalers
        scaler_path = get_scaler_path(city, target_type)
        joblib.dump((feature_scaler, target_scaler), scaler_path)
        
        return model, (feature_scaler, target_scaler), features
    except Exception as e:
        logger.error(f"Error training model for {city} {target_type}: {str(e)}")
        return None, None, None

def load_lstm_model(city, target_type):
    """Load trained LSTM model and scaler"""
    model_path = get_model_path(city, target_type)
    scaler_path = get_scaler_path(city, target_type)
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            model = load_model(model_path, compile=False)  # Load without compilation
            model.compile(optimizer='adam', loss='mean_squared_error')  # Recompile with proper loss
            scalers = joblib.load(scaler_path)
            return model, scalers
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None, None
    return None, None

def forecast_with_lstm(model, scalers, features, last_sequence, forecast_days=5):
    """Make forecasts using LSTM model"""
    try:
        feature_scaler, target_scaler = scalers
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(forecast_days):
            # Reshape sequence for prediction
            X_pred = current_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1])
            
            # Make prediction
            pred = model.predict(X_pred, verbose=0)[0]
            
            # Add prediction to results
            predictions.append(pred[0])
            
            # Update sequence for next prediction
            new_row = current_sequence[-1].copy()
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1] = new_row
        
        # Inverse transform predictions
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = target_scaler.inverse_transform(predictions)
        
        return predictions.flatten()
    except Exception as e:
        logger.error(f"Error making forecasts: {str(e)}")
        return None

def forecast_temperature_lstm(csv_path, city, target_type, forecast_days=5):
    """Main function to forecast temperature using LSTM"""
    try:
        # Try to load existing model
        model, scalers = load_lstm_model(city, target_type)
        
        if model is None:
            # Train new model if none exists
            model, scalers, features = train_lstm_model(csv_path, city, target_type)
            if model is None:
                logger.error(f"Failed to train model for {city} {target_type}")
                return None
        else:
            # Get features from the data
            df = pd.read_csv(csv_path)
            features = ['temp', 'temp_min', 'temp_max', 'humidity', 'pressure', 'wind_speed']
            features = [f for f in features if f in df.columns]
            features.extend(['hour', 'sin_day', 'cos_day'])
        
        feature_scaler, target_scaler = scalers
        
        # Prepare latest data for forecasting
        df = pd.read_csv(csv_path)
        if len(df) < 7:
            logger.error(f"Not enough data for forecasting in {csv_path}")
            return None
            
        df['date'] = pd.to_datetime(df['date'])
        
        # Create time features
        df['hour'] = df['date'].dt.hour
        df['day_of_year'] = df['date'].dt.dayofyear
        df['sin_day'] = np.sin(2 * np.pi * df['day_of_year']/365)
        df['cos_day'] = np.cos(2 * np.pi * df['day_of_year']/365)
        
        # Select and prepare features
        df[features] = df[features].fillna(method='ffill')
        df = df.dropna(subset=features)
        
        if len(df) < 7:
            logger.error(f"Not enough data after cleaning in {csv_path}")
            return None
        
        # Scale the data
        scaled_data = feature_scaler.transform(df[features])
        
        # Get last sequence for forecasting
        last_sequence = scaled_data[-7:]
        
        # Make forecasts
        predictions = forecast_with_lstm(model, scalers, features, last_sequence, forecast_days)
        
        # if predictions is not None:
        #     # Plot forecast
        #     with safe_plot_context():
        #         plt.figure(figsize=(12, 6))
        #         dates = pd.date_range(start=df['date'].iloc[-1], periods=forecast_days+1)[1:]
        #         plt.plot(dates, predictions, marker='o', label='Forecast')
        #         plt.title(f'Temperature Forecast - {city} {target_type}')
        #         plt.xlabel('Date')
        #         plt.ylabel('Temperature')
        #         plt.legend()
        #         plt.xticks(rotation=45)
                
        #         # Save forecast plot
        #         plot_path = os.path.join(MODEL_STORAGE_PATH, f"{city}_{target_type}_forecast.png")
        #         plt.savefig(plot_path, bbox_inches='tight')
        #         plt.close()
        
        return predictions
    except Exception as e:
        logger.error(f"Error in forecast_temperature_lstm for {city} {target_type}: {str(e)}")
        return None 
