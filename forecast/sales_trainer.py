import numpy as np
import torch
from torch.utils.data import DataLoader
from sales_data_processor import SalesDataProcessor, SalesDataset
from sales_lstm_model import SalesForecaster
import threading
import time

class SalesTrainer:
    """Handles training and evaluation of the sales forecasting model"""
    
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.processor = None
        self.forecaster = None
        self.train_loader = None
        self.val_loader = None
        self.is_training = False
        self.training_progress = 0.0
        self.training_status = "Ready"
        self.results = None
        
        # Training parameters (default values)
        self.params = {
            'sequence_length': 8,
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'epochs': 50,
            'batch_size': 16,
            'test_size': 0.2
        }
    
    def update_parameters(self, **kwargs):
        """Update model parameters"""
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value
    
    def prepare_data(self):
        """Prepare and preprocess the data"""
        try:
            self.training_status = "Loading data..."
            
            # Initialize processor
            self.processor = SalesDataProcessor(self.csv_path)
            self.processor.load_and_preprocess()
            
            self.training_status = "Creating time series..."
            
            # Create weekly time series
            weekly_data = self.processor.create_weekly_timeseries()
            
            self.training_status = "Creating sequences..."
            
            # Create sequences
            X, y = self.processor.create_sequences(
                sequence_length=self.params['sequence_length'],
                target_col='Weekly_Quantity'
            )
            
            self.training_status = "Splitting data..."
            
            # Split data
            X_train, X_test, y_train, y_test = self.processor.get_train_test_split(
                test_size=self.params['test_size']
            )
            
            # Create datasets
            train_dataset = SalesDataset(X_train, y_train)
            val_dataset = SalesDataset(X_test, y_test)
            
            # Create data loaders
            self.train_loader = DataLoader(
                train_dataset, 
                batch_size=self.params['batch_size'], 
                shuffle=True
            )
            self.val_loader = DataLoader(
                val_dataset, 
                batch_size=self.params['batch_size'], 
                shuffle=False
            )
            
            self.training_status = "Data prepared"
            return True
            
        except Exception as e:
            self.training_status = f"Error preparing data: {str(e)}"
            print(f"Error preparing data: {e}")
            return False
    
    def initialize_model(self):
        """Initialize the LSTM model"""
        try:
            if self.processor is None:
                self.training_status = "Data not prepared"
                return False
            
            # Get input size
            input_size = len(self.processor.processed_data['feature_cols'])
            
            # Initialize forecaster
            self.forecaster = SalesForecaster(
                input_size=input_size,
                hidden_size=self.params['hidden_size'],
                num_layers=self.params['num_layers'],
                dropout=self.params['dropout'],
                learning_rate=self.params['learning_rate']
            )
            
            self.training_status = "Model initialized"
            return True
            
        except Exception as e:
            self.training_status = f"Error initializing model: {str(e)}"
            print(f"Error initializing model: {e}")
            return False
    
    def train_model_async(self):
        """Train the model in a separate thread"""
        if self.is_training:
            return
        
        def training_thread():
            self.is_training = True
            self.training_progress = 0.0
            
            try:
                self.training_status = "Training started..."
                
                # Train the model
                self.forecaster.train(
                    self.train_loader, 
                    self.val_loader, 
                    epochs=self.params['epochs'],
                    patience=10
                )
                
                self.training_status = "Evaluating model..."
                self.training_progress = 0.9
                
                # Evaluate on test set
                self.evaluate_model()
                
                self.training_progress = 1.0
                self.training_status = "Training completed"
                
            except Exception as e:
                self.training_status = f"Training error: {str(e)}"
                print(f"Training error: {e}")
            finally:
                self.is_training = False
        
        thread = threading.Thread(target=training_thread)
        thread.daemon = True
        thread.start()
    
    def evaluate_model(self):
        """Evaluate the trained model"""
        if self.forecaster is None or self.val_loader is None:
            return None
        
        try:
            # Get test predictions
            all_predictions = []
            all_targets = []
            
            for batch_x, batch_y in self.val_loader:
                predictions = self.forecaster.predict(batch_x.numpy())
                all_predictions.extend(predictions)
                all_targets.extend(batch_y.numpy())
            
            # Convert to numpy arrays
            predictions = np.array(all_predictions)
            targets = np.array(all_targets)
            
            # Denormalize
            predictions_denorm = self.processor.denormalize_target(predictions)
            targets_denorm = self.processor.denormalize_target(targets)
            
            # Calculate metrics
            mae = np.mean(np.abs(targets_denorm - predictions_denorm))
            rmse = np.sqrt(np.mean((targets_denorm - predictions_denorm) ** 2))
            
            # Store results
            self.results = {
                'mae': mae,
                'rmse': rmse,
                'predictions': predictions_denorm,
                'targets': targets_denorm,
                'historical_data': self.processor.weekly_data['Weekly_Quantity'].fillna(0).values,  # Use complete data, fill NaN with 0
                'dates': self.processor.weekly_data['Week_Start'].values
            }
            
            return self.results
            
        except Exception as e:
            self.training_status = f"Evaluation error: {str(e)}"
            print(f"Evaluation error: {e}")
            return None
    
    def generate_forecast(self, weeks_ahead=4):
        """Generate future sales forecast"""
        if self.forecaster is None or self.processor is None:
            return None
        
        try:
            # Get the last sequence for forecasting
            last_sequence = self.processor.processed_data['X'][-1].copy()
            
            # Generate forecast with sequence updating and seasonal patterns
            forecasts = []
            current_sequence = last_sequence.copy()
            
            # Calculate seasonal patterns from historical data
            historical_data = self.processor.processed_data['original_data']['Weekly_Quantity']
            seasonal_pattern = self._calculate_seasonal_pattern(historical_data)
            
            for week in range(weeks_ahead):
                # Make prediction using current sequence
                prediction = self.forecaster.forecast_next_week(current_sequence, self.processor)
                
                # Apply seasonal adjustment if pattern detected
                if seasonal_pattern is not None:
                    seasonal_factor = seasonal_pattern[week % len(seasonal_pattern)]
                    prediction = prediction * seasonal_factor
                
                forecasts.append(prediction)
                
                # Update sequence by shifting and adding prediction with some variation
                if week < weeks_ahead - 1:  # Don't update on last iteration
                    # Shift the sequence
                    current_sequence[:-1] = current_sequence[1:]
                    
                    # Update the last target value with our prediction (normalized)
                    pred_normalized = (prediction - self.processor.target_min) / (self.processor.target_max - self.processor.target_min + 1e-8)
                    current_sequence[-1, 0] = pred_normalized  # Assuming target is first feature
                    
                    # Add more variation to other features to create more realistic sequences
                    # This helps the model generate more varied predictions
                    import random
                    for i in range(1, min(6, current_sequence.shape[1])):  # Update more features
                        # Add larger variation to create more realistic patterns
                        variation = random.uniform(-0.2, 0.2)
                        current_sequence[-1, i] += variation
                        current_sequence[-1, i] = max(0, min(1, current_sequence[-1, i]))  # Keep in [0,1] range
                    
                    # Add some noise to the target value to create variation
                    noise = random.uniform(-0.05, 0.05)
                    current_sequence[-1, 0] += noise
                    current_sequence[-1, 0] = max(0, min(1, current_sequence[-1, 0]))
            
            # Generate forecast dates for weeks 47-54
            # Get the last training week date (week 46)
            train_end_date = self.processor.processed_data['train_data']['Week_Start'].iloc[-1]
            forecast_dates = []
            for i in range(weeks_ahead):
                from datetime import timedelta
                forecast_dates.append(train_end_date + timedelta(weeks=i+1))
            
            # Use complete historical data for plotting (all 54 weeks from original data)
            # Use the original weekly data before lag features were applied
            historical_data = self.processor.weekly_data['Weekly_Quantity'].fillna(0).values
            dates = self.processor.weekly_data['Week_Start'].values
            
            # Get actual test data for comparison (weeks 47-54 from original data)
            test_data = self.processor.weekly_data['Weekly_Quantity'].iloc[46:54].fillna(0).values  # Weeks 47-54
            
            # Debug: Print actual test data values
            print(f"\n=== TEST DATA DEBUG ===")
            print(f"Original data length: {len(self.processor.weekly_data)}")
            print(f"Test data (weeks 47-54) length: {len(test_data)}")
            print(f"Test data values (weeks 47-54): {test_data}")
            print(f"Historical data length: {len(historical_data)}")
            print(f"Historical data last 8 values: {historical_data[-8:]}")
            print(f"Are they the same? {np.array_equal(test_data, historical_data[-8:])}")
            print(f"=== END DEBUG ===\n")
            
            return {
                'forecasts': forecasts,
                'forecast_dates': forecast_dates,
                'historical_data': historical_data,
                'dates': dates,
                'test_data': test_data
            }
            
        except Exception as e:
            print(f"Forecast error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _calculate_seasonal_pattern(self, data):
        """Calculate seasonal pattern from historical data"""
        if len(data) < 12:  # Need at least 12 weeks to detect seasonal patterns
            return None
        
        # Calculate weekly averages to detect patterns
        weekly_means = []
        for i in range(0, len(data), 4):  # Group by 4-week periods
            period_data = data.iloc[i:i+4]
            if len(period_data) > 0:
                weekly_means.append(period_data.mean())
        
        if len(weekly_means) < 3:
            return None
        
        # Calculate seasonal factors (relative to overall mean)
        overall_mean = data.mean()
        seasonal_factors = [mean / overall_mean for mean in weekly_means]
        
        # Normalize factors to be close to 1.0 (don't make them too extreme)
        seasonal_factors = [max(0.7, min(1.3, factor)) for factor in seasonal_factors]
        
        return seasonal_factors
    
    def get_model_info(self):
        """Get model information"""
        if self.forecaster is None:
            return None
        
        return self.forecaster.get_model_info()
    
    def get_training_status(self):
        """Get current training status"""
        return {
            'is_training': self.is_training,
            'progress': self.training_progress,
            'status': self.training_status,
            'results': self.results
        }
    
    def reset(self):
        """Reset the trainer"""
        self.is_training = False
        self.training_progress = 0.0
        self.training_status = "Ready"
        self.results = None
        self.forecaster = None
        self.processor = None
        self.train_loader = None
        self.val_loader = None
