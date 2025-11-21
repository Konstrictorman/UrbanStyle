import numpy as np
import torch
from torch.utils.data import DataLoader
from sales_data_processor import SalesDataProcessor, SalesDataset
from sales_lstm_model import SalesForecaster
import threading
import time

class SalesTrainer:
    """Handles training and evaluation of the sales forecasting model"""
    
    def __init__(self, csv_path, category=None):
        self.csv_path = csv_path
        self.category = category  # Category for category-specific training
        self.processor = None
        self.forecaster = None
        self.train_loader = None
        self.val_loader = None
        self.is_training = False
        self.training_progress = 0.0
        self.training_status = "Ready"
        self.results = None
        self.category_data = None  # Store category-specific weekly data
        self.train_data = None
        self.test_data = None
        self.feature_cols = None
        
        # Training parameters (default values)
        self.params = {
            'sequence_length': 8,
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'epochs': 50,
            'batch_size': 16,
            'test_size': 0.2  # 80/20 split
        }
    
    def update_parameters(self, **kwargs):
        """Update model parameters"""
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value
    
    def prepare_data(self):
        """Prepare and preprocess the data (category-specific if category is set)"""
        try:
            self.training_status = "Loading data..."
            
            # Initialize processor
            self.processor = SalesDataProcessor(self.csv_path)
            self.processor.load_and_preprocess()
            
            self.training_status = "Creating time series..."
            
            if self.category:
                # Category-specific processing
                self.training_status = f"Processing {self.category} category..."
                self.category_data = self.processor.get_category_data(self.category)
                
                self.training_status = "Creating sequences..."
                
                # Create sequences with 80/20 split
                X_train, X_test, y_train, y_test, self.train_data, self.test_data, self.feature_cols = \
                    self.processor.create_sequences_from_data(
                        self.category_data,
                        sequence_length=self.params['sequence_length'],
                        target_col='Weekly_Quantity',
                        test_size=self.params['test_size']  # 0.2 for 80/20 split
                    )
                
                # Create a minimal processed_data structure for forecasting compatibility
                # This is needed because forecast_next_week expects processor.processed_data
                self.processor.processed_data = {
                    'sequence_length': self.params['sequence_length'],
                    'feature_cols': self.feature_cols,
                    'target_col': 'Weekly_Quantity',
                    'X': X_train,  # Store training sequences for reference
                    'y': y_train,
                    'train_data': self.train_data,
                    'test_data': self.test_data,
                    'original_data': self.category_data
                }
            else:
                # All categories combined (original behavior)
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
            
            # Validate that we have enough data
            if len(X_train) == 0:
                self.training_status = "Error: No training sequences created. Not enough data."
                print(f"Error: X_train is empty. X_train shape: {X_train.shape if hasattr(X_train, 'shape') else 'N/A'}")
                return False
            
            # X_test can be empty if test data is too small, but create_sequences_from_data should handle this
            # by creating a validation split from training data. If it's still empty, that's a problem.
            if len(X_test) == 0:
                self.training_status = "Error: No validation sequences available. Cannot train model."
                print(f"Error: X_test is empty after sequence creation. This should not happen.")
                return False
            
            # Create datasets
            train_dataset = SalesDataset(X_train, y_train)
            val_dataset = SalesDataset(X_test, y_test)
            
            # Create data loaders
            # Adjust batch size if we have fewer samples than batch size
            train_batch_size = min(self.params['batch_size'], len(X_train))
            val_batch_size = min(self.params['batch_size'], len(X_test))
            
            self.train_loader = DataLoader(
                train_dataset, 
                batch_size=train_batch_size, 
                shuffle=True
            )
            self.val_loader = DataLoader(
                val_dataset, 
                batch_size=val_batch_size, 
                shuffle=False
            )
            
            print(f"Created data loaders: train batches={len(self.train_loader)}, val batches={len(self.val_loader)}")
            
            self.training_status = "Data prepared"
            return True
            
        except Exception as e:
            self.training_status = f"Error preparing data: {str(e)}"
            print(f"Error preparing data: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def initialize_model(self):
        """Initialize the LSTM model"""
        try:
            if self.processor is None:
                self.training_status = "Data not prepared"
                return False
            
            # Get input size
            if self.category and self.feature_cols is not None:
                # Category-specific: use feature_cols from category data
                input_size = len(self.feature_cols)
            elif self.processor.processed_data is not None:
                # All categories: use feature_cols from processed_data
                input_size = len(self.processor.processed_data['feature_cols'])
            else:
                self.training_status = "Data not processed"
                return False
            
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
            import traceback
            traceback.print_exc()
            return False
    
    def train_model_async(self):
        """Train the model in a separate thread"""
        if self.is_training:
            return
        
        # Check if model is initialized
        if self.forecaster is None:
            self.training_status = "Error: Model not initialized. Please prepare data first."
            print("Error: Cannot train - model not initialized")
            return
        
        # Check if data loaders are ready
        if self.train_loader is None or self.val_loader is None:
            self.training_status = "Error: Data not prepared. Please prepare data first."
            print("Error: Cannot train - data loaders not ready")
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
            
            # Get historical data based on category or all data
            if self.category and self.category_data is not None:
                historical_data = self.category_data['Weekly_Quantity'].fillna(0).values
                dates = self.category_data['Week_Start'].values
            else:
                historical_data = self.processor.weekly_data['Weekly_Quantity'].fillna(0).values
                dates = self.processor.weekly_data['Week_Start'].values
            
            # Store results
            self.results = {
                'mae': mae,
                'rmse': rmse,
                'predictions': predictions_denorm,
                'targets': targets_denorm,
                'historical_data': historical_data,
                'dates': dates
            }
            
            return self.results
            
        except Exception as e:
            self.training_status = f"Evaluation error: {str(e)}"
            print(f"Evaluation error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_forecast(self, weeks_ahead=4):
        """Generate future sales forecast (4 weeks ahead per category)"""
        if self.forecaster is None or self.processor is None:
            return None
        
        try:
            # Get the last sequence for forecasting
            if self.category and self.train_data is not None and self.processor.processed_data is not None:
                # Category-specific: get last sequence from original (non-normalized) training data
                # forecast_next_week expects original scale data and will normalize it internally
                train_features = self.train_data[self.feature_cols].values.astype(np.float64)
                seq_len = self.params['sequence_length']
                # Get the last sequence_length weeks from training data (original scale)
                last_sequence = train_features[-seq_len:].copy()
                
                # Get historical data and dates (use full category_data, not just train_data)
                historical_data = self.category_data['Weekly_Quantity'].fillna(0).values
                dates = self.category_data['Week_Start'].values
                
                # Get test data for comparison
                test_data = self.test_data['Weekly_Quantity'].fillna(0).values if self.test_data is not None else None
                
                # Get last training week date (end of training period, not end of all data)
                train_end_date = self.train_data['Week_Start'].iloc[-1]
            else:
                # All categories: use processed_data
                if self.processor.processed_data is None:
                    print("Error: processed_data is None")
                    return None
                last_sequence = self.processor.processed_data['X'][-1].copy()
                historical_data = self.processor.weekly_data['Weekly_Quantity'].fillna(0).values
                dates = self.processor.weekly_data['Week_Start'].values
                train_end_date = self.processor.processed_data['train_data']['Week_Start'].iloc[-1]
                test_data = None
            
            # Generate forecast with sequence updating and seasonal patterns
            forecasts = []
            current_sequence = last_sequence.copy()
            
            # Calculate seasonal patterns from historical data
            if self.category and self.category_data is not None:
                hist_series = self.category_data['Weekly_Quantity']
            else:
                hist_series = self.processor.processed_data['original_data']['Weekly_Quantity']
            
            seasonal_pattern = self._calculate_seasonal_pattern(hist_series)
            
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
                    
                    # Update the last target value with our prediction (in original scale)
                    # The sequence is in original scale, so we put the prediction directly
                    current_sequence[-1, 0] = prediction  # Assuming target is first feature
                    
                    # Update other features based on the prediction and historical patterns
                    # We need to estimate what other features would be for the next week
                    import random
                    # Get the previous week's features as a base (from the second-to-last time step)
                    if current_sequence.shape[0] > 1:
                        prev_features = current_sequence[-2, :].copy()
                        prev_prediction = current_sequence[-2, 0]  # Previous week's target value
                        pred_change = prediction - prev_prediction
                        
                        # Update other features with small variations based on prediction trend
                        for i in range(1, current_sequence.shape[1]):
                            # Add variation proportional to the prediction change
                            variation = random.uniform(-abs(pred_change) * 0.1, abs(pred_change) * 0.1)
                            current_sequence[-1, i] = prev_features[i] + variation
                            # Keep values within reasonable bounds (don't normalize, just clip extreme values)
                            if hasattr(self.processor, 'feature_min') and self.processor.feature_min is not None and len(self.processor.feature_min) > i:
                                feature_min_val = self.processor.feature_min[i]
                                feature_max_val = self.processor.feature_max[i] if hasattr(self.processor, 'feature_max') and len(self.processor.feature_max) > i else prev_features[i] * 2
                                current_sequence[-1, i] = max(feature_min_val, min(feature_max_val, current_sequence[-1, i]))
                            else:
                                # Fallback: keep within reasonable range
                                current_sequence[-1, i] = max(0, current_sequence[-1, i])
                    
                    # Add small noise to the target value to create variation
                    noise = random.uniform(-prediction * 0.05, prediction * 0.05)
                    current_sequence[-1, 0] = max(0, current_sequence[-1, 0] + noise)
            
            # Generate forecast dates for next 4 weeks
            from datetime import timedelta
            forecast_dates = []
            for i in range(weeks_ahead):
                forecast_dates.append(train_end_date + timedelta(weeks=i+1))
            
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
        
        # Prevent division by zero - if overall_mean is 0 or very small, return None
        if overall_mean == 0 or abs(overall_mean) < 1e-8:
            return None
        
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
