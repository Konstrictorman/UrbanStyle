# Sales Forecasting with LSTM - Pygame Interface

## 🎯 **Overview**

This implementation provides a comprehensive sales forecasting system using LSTM neural networks with an interactive pygame interface. The system processes retail sales data from the Kaggle dataset and provides weekly sales predictions to support inventory decisions.

## 📊 **Features**

### **Core Functionality**

- **Weekly Time Series Processing**: Aggregates daily transactions into weekly sales data
- **LSTM Model**: Neural network architecture for sequence-to-sequence prediction
- **Interactive Interface**: Pygame-based GUI with sliders and buttons
- **Real-time Training**: Asynchronous model training with progress tracking
- **Forecast Visualization**: Professional plotting of historical data and predictions
- **Model Evaluation**: MAE and RMSE metrics for performance assessment

### **Interactive Controls**

- **Parameter Sliders**: Adjust model hyperparameters in real-time

  - Sequence Length (4-12 weeks)
  - Hidden Size (32-128 neurons)
  - LSTM Layers (1-4 layers)
  - Dropout Rate (0.0-0.5)
  - Learning Rate (0.0001-0.01)
  - Training Epochs (10-100)

- **Action Buttons**:
  - **Prepare Data**: Load and preprocess the Kaggle dataset
  - **Train Model**: Start LSTM training with current parameters
  - **Generate Forecast**: Create and visualize sales predictions

## 🏗️ **Architecture**

### **Components**

1. **`sales_data_processor.py`**: Data preprocessing and time series creation
2. **`sales_lstm_model.py`**: LSTM neural network implementation
3. **`slider.py`**: Interactive pygame slider components
4. **`forecast_plotter.py`**: Professional plotting and visualization
5. **`sales_trainer.py`**: Training orchestration and evaluation
6. **`sales_forecast_app.py`**: Main pygame application

### **Data Flow**

```
Raw CSV Data → Weekly Aggregation → Feature Engineering →
LSTM Training → Model Evaluation → Forecast Generation → Visualization
```

## 📈 **Model Architecture**

### **LSTM Configuration**

- **Input**: 8-week sequences with 42 features
- **Hidden Layers**: 2 LSTM layers with 64 neurons
- **Dropout**: 0.2 for regularization
- **Output**: Single prediction for next week's sales
- **Parameters**: ~61,000 trainable parameters

### **Features Used**

- Historical sales quantities and revenue
- Product category distributions
- Customer demographics (age, gender)
- Time-based features (week, month, quarter)
- Lag features (1-8 weeks)
- Rolling averages (4 and 8 weeks)

## 🎮 **Usage Instructions**

### **Prerequisites**

```bash
pip install -r requirements.txt
```

### **Running the Application**

```bash
python sales_forecast_app.py
```

### **Step-by-Step Process**

1. **Adjust Parameters**: Use sliders to set model hyperparameters
2. **Prepare Data**: Click "Prepare Data" to load and preprocess the dataset
3. **Train Model**: Click "Train Model" to start LSTM training
4. **Generate Forecast**: Click "Generate Forecast" to see predictions
5. **Analyze Results**: View MAE/RMSE metrics and forecast visualization

## 📊 **Performance Metrics**

### **Evaluation Results**

- **MAE (Mean Absolute Error)**: ~13-18 units
- **RMSE (Root Mean Squared Error)**: ~18-23 units
- **Training Time**: ~10-20 epochs with early stopping
- **Model Size**: 61,000 parameters

### **Data Statistics**

- **Total Transactions**: 1,000 records
- **Time Period**: Full year 2023 (54 weeks)
- **Product Categories**: 5 categories (Electronics, Clothing, Beauty, Baby Stuff, Sports)
- **Sequence Length**: 8 weeks for prediction
- **Feature Count**: 42 engineered features

## 🔧 **Technical Details**

### **Data Preprocessing**

- **Weekly Aggregation**: Daily transactions grouped by week
- **Feature Engineering**: 42 features including lags, trends, and demographics
- **Normalization**: Min-max scaling for stable training
- **Train/Test Split**: 80/20 chronological split

### **Model Training**

- **Optimizer**: Adam with learning rate scheduling
- **Loss Function**: Mean Squared Error
- **Regularization**: Dropout and gradient clipping
- **Early Stopping**: Prevents overfitting with patience=10

### **Forecasting**

- **Horizon**: 4-8 weeks ahead
- **Method**: Sequential prediction using last known sequence
- **Uncertainty**: Simplified approach (real implementation would include confidence intervals)

## 🎨 **Interface Design**

### **Layout**

- **Left Panel**: Parameter sliders and control buttons
- **Right Panel**: Forecast visualization with historical data
- **Status Area**: Training progress and evaluation metrics
- **Progress Bar**: Real-time training progress indicator

### **Visual Elements**

- **Color Scheme**: Professional blue/gray palette
- **Interactive Sliders**: Real-time parameter adjustment
- **Forecast Plot**: Historical data (blue) vs. predictions (red)
- **Metrics Display**: MAE and RMSE prominently shown

## 🚀 **Business Applications**

### **Inventory Management**

- **Demand Forecasting**: Predict weekly sales quantities
- **Stock Planning**: Optimize inventory levels
- **Seasonal Planning**: Identify peak and low periods
- **Cost Reduction**: Minimize overstock and stockouts

### **Strategic Planning**

- **Trend Analysis**: Understand sales patterns
- **Performance Monitoring**: Track forecasting accuracy
- **Decision Support**: Data-driven inventory decisions
- **Resource Allocation**: Plan staffing and resources

## 🔮 **Future Enhancements**

### **Model Improvements**

- **Multi-target Prediction**: Predict both quantity and revenue
- **Confidence Intervals**: Provide uncertainty estimates
- **Ensemble Methods**: Combine multiple model predictions
- **Advanced Features**: External factors (weather, promotions)

### **Interface Enhancements**

- **Real-time Updates**: Live parameter adjustment during training
- **Export Functionality**: Save forecasts and models
- **Comparison Tools**: Multiple model comparison
- **Interactive Charts**: Zoom and pan functionality

## 📝 **Implementation Notes**

### **Key Design Decisions**

- **Weekly Aggregation**: Balances detail with computational efficiency
- **Feature Engineering**: Comprehensive feature set for robust predictions
- **Asynchronous Training**: Non-blocking UI during model training
- **Modular Architecture**: Clean separation of concerns

### **Performance Optimizations**

- **Gradient Clipping**: Prevents exploding gradients
- **Early Stopping**: Reduces overfitting
- **Data Normalization**: Improves training stability
- **Batch Processing**: Efficient training with mini-batches

This implementation successfully combines machine learning with interactive visualization to create a powerful tool for retail sales forecasting and inventory management.
