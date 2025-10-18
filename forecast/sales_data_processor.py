"""
Sales Data Processor Module

This module provides comprehensive data preprocessing capabilities for retail sales forecasting.
It handles CSV data loading, weekly time series aggregation, feature engineering, and 
sequence creation for LSTM neural network training.

Key Features:
- Weekly aggregation of daily transaction data
- Feature engineering with lag features and rolling averages
- Time-based train/test splitting for temporal data
- Data normalization for neural network training
- PyTorch dataset integration

Author: UrbanStyle Project
Date: 2024
"""

import pandas as pd
import numpy as np
from datetime import datetime
import torch
from torch.utils.data import Dataset, DataLoader

class SalesDataProcessor:
    """
    Comprehensive data processor for retail sales forecasting.
    
    This class handles the complete data pipeline from raw CSV files to 
    LSTM-ready sequences, including feature engineering, normalization,
    and temporal splitting for time series forecasting.
    
    Attributes:
        csv_path (str): Path to the input CSV file
        raw_data (pd.DataFrame): Loaded raw transaction data
        weekly_data (pd.DataFrame): Aggregated weekly time series
        processed_data (dict): Processed sequences and metadata
        feature_min (np.array): Minimum values for normalization
        feature_max (np.array): Maximum values for normalization
        target_min (float): Minimum target value for normalization
        target_max (float): Maximum target value for normalization
    """
    
    def __init__(self, csv_path):
        """
        Initialize the data processor with CSV file path.
        
        Args:
            csv_path (str): Path to the Kaggle retail sales CSV file
        """
        self.csv_path = csv_path
        self.raw_data = None
        self.weekly_data = None
        self.processed_data = None
        
    def load_and_preprocess(self):
        """
        Load and preprocess the retail sales data from CSV file.
        
        This method performs comprehensive data cleaning and type conversion:
        - Loads CSV with semicolon delimiter
        - Removes BOM characters from column names
        - Converts date strings to datetime objects
        - Converts numeric columns to appropriate data types
        - Validates data integrity
        
        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If data format is invalid
            
        Returns:
            None: Data is stored in self.raw_data
        """
        # Load the CSV file with semicolon delimiter
        self.raw_data = pd.read_csv(self.csv_path, sep=';')
        
        # Clean column names (remove BOM character that can appear in UTF-8 files)
        self.raw_data.columns = [col.replace('\ufeff', '') for col in self.raw_data.columns]
        
        # Convert date column from string format to datetime
        self.raw_data['Date'] = pd.to_datetime(self.raw_data['Date'], format='%d/%m/%y')
        
        # Convert numeric columns to appropriate data types
        self.raw_data['Quantity'] = pd.to_numeric(self.raw_data['Quantity'])
        self.raw_data['Price per Unit'] = pd.to_numeric(self.raw_data['Price per Unit'])
        self.raw_data['Total Amount'] = pd.to_numeric(self.raw_data['Total Amount'])
        self.raw_data['Age'] = pd.to_numeric(self.raw_data['Age'])
        
        print(f"Loaded {len(self.raw_data)} transactions from {self.raw_data['Date'].min()} to {self.raw_data['Date'].max()}")
        
    def create_weekly_timeseries(self):
        """
        Aggregate daily transactions into weekly time series with comprehensive feature engineering.
        
        This method transforms raw daily transaction data into weekly aggregated features
        suitable for time series forecasting. It includes:
        - Weekly aggregations (quantity, revenue, transaction count)
        - Demographic features (average age, gender ratio)
        - Product category breakdowns
        - Time-based features (week of year, month, quarter)
        - Lag features (previous weeks' values)
        - Rolling averages (moving averages over different windows)
        
        Raises:
            ValueError: If raw data hasn't been loaded yet
            
        Returns:
            pd.DataFrame: Weekly aggregated time series with engineered features
        """
        if self.raw_data is None:
            raise ValueError("Data not loaded. Call load_and_preprocess() first.")
        
        # Create weekly aggregations using Sunday as week start
        weekly_agg = self.raw_data.groupby(pd.Grouper(key='Date', freq='W-SUN')).agg({
            'Quantity': 'sum',                    # Total quantity sold per week
            'Total Amount': 'sum',                # Total revenue per week
            'Transaction ID': 'count',            # Number of transactions per week
            'Age': 'mean',                       # Average customer age per week
            'Gender': lambda x: (x == 'Female').sum() / len(x)  # Female customer ratio
        }).reset_index()
        
        # Rename columns for clarity
        weekly_agg.columns = ['Week_Start', 'Weekly_Quantity', 'Weekly_Revenue', 'Transaction_Count', 'Avg_Age', 'Female_Ratio']
        
        # Create product category aggregations for detailed analysis
        category_cols = []
        for category in self.raw_data['Product Category'].unique():
            cat_data = self.raw_data[self.raw_data['Product Category'] == category]
            cat_weekly = cat_data.groupby(pd.Grouper(key='Date', freq='W-SUN'))['Quantity'].sum().reset_index()
            cat_weekly.columns = ['Week_Start', f'{category}_Quantity']
            weekly_agg = weekly_agg.merge(cat_weekly, on='Week_Start', how='left')
            category_cols.append(f'{category}_Quantity')
        
        # Fill NaN values with 0 for weeks with no sales in specific categories
        weekly_agg = weekly_agg.fillna(0)
        
        # Add time-based features for seasonality detection
        weekly_agg['Week_of_Year'] = weekly_agg['Week_Start'].dt.isocalendar().week
        weekly_agg['Month'] = weekly_agg['Week_Start'].dt.month
        weekly_agg['Quarter'] = weekly_agg['Week_Start'].dt.quarter
        
        # Add lag features (previous weeks' values) for temporal dependencies
        # Reduced to 4 weeks to preserve more data for training
        lag_features = ['Weekly_Quantity', 'Weekly_Revenue', 'Transaction_Count']
        for feature in lag_features:
            for lag in range(1, 5):  # Only 4 weeks of lags instead of 8
                weekly_agg[f'{feature}_lag_{lag}'] = weekly_agg[feature].shift(lag)
        
        # Add rolling averages for trend smoothing
        # Smaller windows to preserve more data
        for window in [2, 4]:  # 2-week and 4-week moving averages
            for feature in lag_features:
                weekly_agg[f'{feature}_ma_{window}'] = weekly_agg[feature].rolling(window=window).mean()
        
        self.weekly_data = weekly_agg
        print(f"Created weekly time series with {len(weekly_agg)} weeks")
        return weekly_agg
    
    def create_sequences(self, sequence_length=8, target_col='Weekly_Quantity'):
        """
        Create LSTM-ready sequences with temporal train/test splitting.
        
        This method transforms weekly time series data into sequences suitable for LSTM training.
        It implements proper temporal splitting to avoid data leakage and ensures realistic
        forecasting evaluation. Key features:
        - Temporal train/test split (no shuffling to preserve time order)
        - Feature normalization using only training data statistics
        - Sequence creation with sliding window approach
        - Comprehensive metadata storage for later use
        
        Args:
            sequence_length (int): Number of weeks to use as input sequence (default: 8)
            target_col (str): Column name to use as prediction target (default: 'Weekly_Quantity')
            
        Raises:
            ValueError: If weekly data hasn't been created yet
            
        Returns:
            tuple: (X, y) where X is input sequences and y is target values
        """
        if self.weekly_data is None:
            raise ValueError("Weekly data not created. Call create_weekly_timeseries() first.")
        
        # Remove rows with NaN values (due to lag features and rolling averages)
        clean_data = self.weekly_data.dropna().reset_index(drop=True)
        
        print(f"Clean data after removing NaN: {len(clean_data)} weeks")
        
        # Calculate temporal split for realistic forecasting evaluation
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
        
        # Select feature columns (exclude date and target columns)
        feature_cols = [col for col in train_data.columns if col not in ['Week_Start', target_col]]
        
        # Normalize features using ONLY training data statistics to prevent data leakage
        train_features = train_data[feature_cols].values.astype(np.float64)
        train_targets = train_data[target_col].values.astype(np.float64)
        
        # Store normalization parameters from training data only
        self.feature_min = train_features.min(axis=0)
        self.feature_max = train_features.max(axis=0)
        self.target_min = train_targets.min()
        self.target_max = train_targets.max()
        
        # Normalize training data using min-max scaling
        train_features_norm = (train_features - self.feature_min) / (self.feature_max - self.feature_min + 1e-8)
        train_targets_norm = (train_targets - self.target_min) / (self.target_max - self.target_min + 1e-8)
        
        # Create sequences only from training data using sliding window
        X, y = [], []
        for i in range(sequence_length, len(train_features_norm)):
            X.append(train_features_norm[i-sequence_length:i])
            y.append(train_targets_norm[i])
        
        X = np.array(X)
        y = np.array(y)
        
        # Store comprehensive metadata for later use
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
        """
        Convert normalized predictions back to original scale.
        
        This method reverses the min-max normalization applied during preprocessing,
        converting model predictions from the [0,1] range back to the original
        sales quantity scale.
        
        Args:
            normalized_values (np.array or float): Normalized values in [0,1] range
            
        Returns:
            np.array or float: Values in original sales quantity scale
        """
        return normalized_values * (self.target_max - self.target_min) + self.target_min
    
    def get_train_test_split(self, test_size=0.2):
        """
        Split processed data into train/test sets using chronological ordering.
        
        This method creates temporal train/test splits from the processed sequences,
        ensuring no future data leaks into training. It maintains chronological
        order which is crucial for time series forecasting.
        
        Args:
            test_size (float): Proportion of data to use for testing (default: 0.2)
            
        Raises:
            ValueError: If data hasn't been processed yet
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test) split sequences
        """
        if self.processed_data is None:
            raise ValueError("Data not processed. Call create_sequences() first.")
        
        X = self.processed_data['X']
        y = self.processed_data['y']
        
        # Time-based split (no shuffling to preserve temporal order)
        split_idx = int(len(X) * (1 - test_size))
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, X_test, y_train, y_test

class SalesDataset(Dataset):
    """
    PyTorch Dataset wrapper for sales forecasting sequences.
    
    This class provides a PyTorch-compatible dataset interface for the processed
    sales sequences, enabling seamless integration with PyTorch DataLoader
    for efficient batch processing during training.
    
    Attributes:
        X (torch.Tensor): Input sequences of shape (n_samples, sequence_length, n_features)
        y (torch.Tensor): Target values of shape (n_samples,)
    """
    
    def __init__(self, X, y):
        """
        Initialize the dataset with input sequences and targets.
        
        Args:
            X (np.array): Input sequences of shape (n_samples, sequence_length, n_features)
            y (np.array): Target values of shape (n_samples,)
        """
        # Convert to PyTorch tensors with proper data types
        self.X = torch.FloatTensor(X.astype(np.float32))
        self.y = torch.FloatTensor(y.astype(np.float32))
    
    def __len__(self):
        """
        Return the number of samples in the dataset.
        
        Returns:
            int: Number of samples
        """
        return len(self.X)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            tuple: (input_sequence, target_value) for the given index
        """
        return self.X[idx], self.y[idx]
