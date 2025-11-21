import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sales_data_processor import SalesDataProcessor
from sales_lstm_model import SalesLSTM
from sales_trainer import SalesTrainer

class MultiCategoryTrainer:
    """Trainer for multiple product categories"""
    
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.processor = SalesDataProcessor(csv_path)
        self.processor.load_and_preprocess()
        self.categories = self.processor.get_categories()
        self.trainers = {}
        
        # Initialize trainers for each category
        for category in self.categories:
            self.trainers[category] = SalesTrainer(csv_path, category)
    
    def prepare_data(self, sequence_length=8, hidden_size=64, num_layers=2, 
                    dropout_rate=0.2, learning_rate=0.001, epochs=50):
        """Prepare data for all categories"""
        results = {}
        
        for category in self.categories:
            print(f"\nPreparing data for {category}...")
            try:
                # Get category-specific data
                category_data = self.processor.get_category_data(category)
                
                # Create sequences for this category
                X, y = self.processor.create_sequences_from_data(category_data, sequence_length)
                
                # Store in trainer
                self.trainers[category].processor = self.processor  # Use shared processor
                self.trainers[category].processor.weekly_data = category_data
                
                # Get feature columns from the data
                feature_cols = [col for col in category_data.columns if col not in ['Week_Start', 'Weekly_Quantity']]
                
                # Split data for training
                split_idx = int(len(X) * 0.8)  # 80% for training
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
                
                train_data = category_data.iloc[:split_idx + sequence_length].copy()
                test_data = category_data.iloc[split_idx + sequence_length:].copy()
                
                self.trainers[category].processor.processed_data = {
                    'X': X, 'y': y, 'original_data': category_data, 'feature_cols': feature_cols,
                    'sequence_length': sequence_length, 'target_col': 'Weekly_Quantity',
                    'train_data': train_data, 'test_data': test_data
                }
                
                results[category] = {
                    'status': 'success',
                    'sequences': len(X),
                    'weeks': len(category_data)
                }
                
            except Exception as e:
                print(f"Error preparing data for {category}: {e}")
                results[category] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return results
    
    def train_all_models(self, sequence_length=8, hidden_size=64, num_layers=2, 
                        dropout_rate=0.2, learning_rate=0.001, epochs=50):
        """Train models for all categories"""
        results = {}
        
        for category in self.categories:
            print(f"\nTraining model for {category}...")
            try:
                # Update trainer parameters
                self.trainers[category].sequence_length = sequence_length
                self.trainers[category].hidden_size = hidden_size
                self.trainers[category].num_layers = num_layers
                self.trainers[category].dropout_rate = dropout_rate
                self.trainers[category].learning_rate = learning_rate
                self.trainers[category].epochs = epochs
                
                # Create data loaders for this category
                from torch.utils.data import DataLoader
                from sales_data_processor import SalesDataset
                
                # Get category data
                category_data = self.processor.get_category_data(category)
                
                # Split data for this category
                X_train, X_test, y_train, y_test = self._split_category_data(category_data, sequence_length)
                
                # Create datasets and loaders
                train_dataset = SalesDataset(X_train, y_train)
                val_dataset = SalesDataset(X_test, y_test)
                
                train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
                
                # Initialize and train the model
                self.trainers[category].initialize_model()
                self.trainers[category].forecaster.train(
                    train_loader,
                    val_loader,
                    epochs=epochs,
                    patience=10
                )
                
                results[category] = {
                    'status': 'success',
                    'message': f'Model trained for {category}'
                }
                
            except Exception as e:
                print(f"Error training model for {category}: {e}")
                results[category] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return results
    
    def generate_forecasts(self, weeks_ahead=8):
        """Generate forecasts for all categories"""
        forecasts = {}
        
        for category in self.categories:
            try:
                forecast_data = self.trainers[category].generate_forecast(weeks_ahead)
                if forecast_data:
                    forecasts[category] = forecast_data
                else:
                    forecasts[category] = None
            except Exception as e:
                print(f"Error generating forecast for {category}: {e}")
                forecasts[category] = None
        
        return forecasts
    
    def get_training_status(self):
        """Get training status for all categories"""
        status = {}
        
        for category in self.categories:
            try:
                status[category] = self.trainers[category].get_training_status()
            except Exception as e:
                status[category] = {
                    'training': False,
                    'progress': 0.0,
                    'results': None,
                    'error': str(e)
                }
        
        return status
    
    def get_training_status(self):
        """Get overall training status"""
        overall_status = {
            'is_training': False,
            'progress': 0.0,
            'results': None,
            'error': None
        }
        
        # Check if any category is training
        for category in self.categories:
            try:
                status = self.trainers[category].get_training_status()
                if status['is_training']:
                    overall_status['is_training'] = True
                    break
            except:
                pass
        
        return overall_status
    
    def _split_category_data(self, category_data, sequence_length, test_size=0.2):
        """Split category data into train/test sets"""
        # Create sequences for this category
        X, y = self.processor.create_sequences_from_data(category_data, sequence_length)
        
        # Split data
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, X_test, y_train, y_test
