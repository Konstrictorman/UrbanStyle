import pandas as pd
import numpy as np
from datetime import datetime
import torch
from torch.utils.data import Dataset, DataLoader

class SalesDataProcessor:
    def __init__(self, csv_path):
        """Initialize with path to the Kaggle CSV file"""
        self.csv_path = csv_path
        self.raw_data = None
        self.weekly_data = None
        self.processed_data = None
        
    def load_and_preprocess(self):
        """Load and preprocess the retail sales data"""
        # Load the CSV file
        self.raw_data = pd.read_csv(self.csv_path, sep=';')
        
        # Clean column names (remove BOM character)
        self.raw_data.columns = [col.replace('\ufeff', '') for col in self.raw_data.columns]
        
        # Convert date column
        self.raw_data['Date'] = pd.to_datetime(self.raw_data['Date'], format='%d/%m/%y')
        
        # Convert numeric columns
        self.raw_data['Quantity'] = pd.to_numeric(self.raw_data['Quantity'])
        self.raw_data['Price per Unit'] = pd.to_numeric(self.raw_data['Price per Unit'])
        self.raw_data['Total Amount'] = pd.to_numeric(self.raw_data['Total Amount'])
        self.raw_data['Age'] = pd.to_numeric(self.raw_data['Age'])
        
        print(f"Loaded {len(self.raw_data)} transactions from {self.raw_data['Date'].min()} to {self.raw_data['Date'].max()}")
        
    def create_weekly_timeseries(self):
        """Aggregate daily transactions into weekly time series"""
        if self.raw_data is None:
            raise ValueError("Data not loaded. Call load_and_preprocess() first.")
        
        # Create weekly aggregations
        weekly_agg = self.raw_data.groupby(pd.Grouper(key='Date', freq='W-SUN')).agg({
            'Quantity': 'sum',
            'Total Amount': 'sum',
            'Transaction ID': 'count',
            'Age': 'mean',
            'Gender': lambda x: (x == 'Female').sum() / len(x)  # Female ratio
        }).reset_index()
        
        # Rename columns
        weekly_agg.columns = ['Week_Start', 'Weekly_Quantity', 'Weekly_Revenue', 'Transaction_Count', 'Avg_Age', 'Female_Ratio']
        
        # Create product category aggregations
        category_cols = []
        for category in self.raw_data['Product Category'].unique():
            cat_data = self.raw_data[self.raw_data['Product Category'] == category]
            cat_weekly = cat_data.groupby(pd.Grouper(key='Date', freq='W-SUN'))['Quantity'].sum().reset_index()
            cat_weekly.columns = ['Week_Start', f'{category}_Quantity']
            weekly_agg = weekly_agg.merge(cat_weekly, on='Week_Start', how='left')
            category_cols.append(f'{category}_Quantity')
        
        # Fill NaN values with 0
        weekly_agg = weekly_agg.fillna(0)
        
        # Add time-based features
        weekly_agg['Week_of_Year'] = weekly_agg['Week_Start'].dt.isocalendar().week
        weekly_agg['Month'] = weekly_agg['Week_Start'].dt.month
        weekly_agg['Quarter'] = weekly_agg['Week_Start'].dt.quarter
        
        # Add lag features (previous weeks) - reduce lags to preserve more data
        lag_features = ['Weekly_Quantity', 'Weekly_Revenue', 'Transaction_Count']
        for feature in lag_features:
            for lag in range(1, 5):  # Only 4 weeks of lags instead of 8
                weekly_agg[f'{feature}_lag_{lag}'] = weekly_agg[feature].shift(lag)
        
        # Add rolling averages - reduce windows to preserve more data
        for window in [2, 4]:  # Smaller windows
            for feature in lag_features:
                weekly_agg[f'{feature}_ma_{window}'] = weekly_agg[feature].rolling(window=window).mean()
        
        self.weekly_data = weekly_agg
        print(f"Created weekly time series with {len(weekly_agg)} weeks")
        return weekly_agg
    
    def create_sequences(self, sequence_length=8, target_col='Weekly_Quantity'):
        """Create sequences for LSTM training with train/test split at week 46"""
        if self.weekly_data is None:
            raise ValueError("Weekly data not created. Call create_weekly_timeseries() first.")
        
        # Remove rows with NaN values (due to lags and rolling averages)
        clean_data = self.weekly_data.dropna().reset_index(drop=True)
        
        print(f"Clean data after removing NaN: {len(clean_data)} weeks")
        
        # For the desired split (weeks 1-46 training, 47-54 testing), we need to map to the clean data
        # Since lag features shift the data, we need to find the equivalent split in clean_data
        
        # We want to use as much data as possible for training while keeping 8 weeks for testing
        total_weeks = len(clean_data)
        
        # Calculate the split: use 46 weeks for training, 8 weeks for testing
        # But adjust based on available clean data
        if total_weeks >= 54:  # We have enough data for the ideal split
            train_weeks = 46
            test_weeks = 8
        else:  # Adjust proportionally
            # Use 85% for training, 15% for testing (roughly 46/54 ratio)
            train_weeks = int(total_weeks * 0.85)
            test_weeks = total_weeks - train_weeks
        
        train_data = clean_data.iloc[:train_weeks].copy()  # Training data
        test_data = clean_data.iloc[train_weeks:].copy()   # Test data
        
        print(f"Training data: weeks 1-{train_weeks} ({len(train_data)} weeks)")
        print(f"Test data: weeks {train_weeks+1}-{total_weeks} ({len(test_data)} weeks)")
        print(f"Note: Due to lag features, this represents the equivalent of weeks 1-46 and 47-54 in original data")
        
        # Select feature columns
        feature_cols = [col for col in train_data.columns if col not in ['Week_Start', target_col]]
        
        # Normalize features using ONLY training data statistics
        train_features = train_data[feature_cols].values.astype(np.float64)
        train_targets = train_data[target_col].values.astype(np.float64)
        
        # Store normalization parameters from training data only
        self.feature_min = train_features.min(axis=0)
        self.feature_max = train_features.max(axis=0)
        self.target_min = train_targets.min()
        self.target_max = train_targets.max()
        
        # Normalize training data
        train_features_norm = (train_features - self.feature_min) / (self.feature_max - self.feature_min + 1e-8)
        train_targets_norm = (train_targets - self.target_min) / (self.target_max - self.target_min + 1e-8)
        
        # Create sequences only from training data
        X, y = [], []
        for i in range(sequence_length, len(train_features_norm)):
            X.append(train_features_norm[i-sequence_length:i])
            y.append(train_targets_norm[i])
        
        X = np.array(X)
        y = np.array(y)
        
        self.processed_data = {
            'X': X,
            'y': y,
            'feature_cols': feature_cols,
            'target_col': target_col,
            'sequence_length': sequence_length,
            'original_data': clean_data,      # Full dataset for plotting
            'train_data': train_data,         # Training subset (weeks 1-46)
            'test_data': test_data,           # Test subset (weeks 47-54)
            'train_start_week': 1,
            'train_end_week': train_weeks,
            'test_start_week': train_weeks + 1,
            'test_end_week': total_weeks
        }
        
        print(f"Created {len(X)} training sequences of length {sequence_length}")
        return X, y
    
    def denormalize_target(self, normalized_values):
        """Convert normalized predictions back to original scale"""
        return normalized_values * (self.target_max - self.target_min) + self.target_min
    
    def get_train_test_split(self, test_size=0.2):
        """Split data into train/test sets (chronological)"""
        if self.processed_data is None:
            raise ValueError("Data not processed. Call create_sequences() first.")
        
        X = self.processed_data['X']
        y = self.processed_data['y']
        
        # Time-based split (no shuffling)
        split_idx = int(len(X) * (1 - test_size))
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, X_test, y_train, y_test

class SalesDataset(Dataset):
    """PyTorch Dataset for sales forecasting"""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X.astype(np.float32))
        self.y = torch.FloatTensor(y.astype(np.float32))
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
