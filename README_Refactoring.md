# Project Refactoring Summary

## Overview

The project has been successfully refactored to improve organization and maintainability by separating concerns into logical folders.

## New Folder Structure

### 📁 `forecast/` - Sales Forecasting System

Contains all files related to the LSTM-based sales forecasting application:

- **`sales_forecast_app.py`** - Main application with Pygame GUI
- **`sales_data_processor.py`** - Data preprocessing and feature engineering
- **`sales_lstm_model.py`** - LSTM neural network model definition
- **`sales_trainer.py`** - Training and evaluation logic
- **`forecast_plotter.py`** - Single-category plotting functionality
- **`multi_category_trainer.py`** - Multi-category training management
- **`multi_category_plotter.py`** - Multi-category plotting (separate plots)
- **`single_plot_multi_category_plotter.py`** - Multi-category plotting (single plot)
- **`checkbox.py`** - Checkbox UI component for category selection

### 📁 `common/` - Shared UI Components

Contains reusable UI components used across different applications:

- **`button.py`** - Button widget with click handling
- **`slider.py`** - Slider widget with value adjustment
- **`inputfield.py`** - Text input field with validation
- **`progressbar.py`** - Progress bar for loading states

### 📁 Root Directory - Core Applications

Contains the main applications and core files:

- **`store.py`** - Q-learning store simulation with customer movement
- **`sim.py`** - Discrete event simulation for supply chain
- **`data.py`** - Data structures for simulation
- **`forecast.py`** - Original sine wave forecasting (legacy)
- **`tracker.py`** - Performance tracking utilities
- **`assets/`** - Data files (CSV, images, etc.)
- **`requirements.txt`** - Python dependencies

## Import Updates

### Sales Forecast Application

```python
# Updated imports in forecast/sales_forecast_app.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.slider import SliderPanel
from common.button import Button
```

### Store Simulation

```python
# Updated imports in store.py
from common.button import Button
from common.progressbar import ProgressBar
from common.inputfield import InputField
```

## Benefits of Refactoring

1. **Better Organization**: Related files are grouped together logically
2. **Reusability**: Common UI components can be easily shared
3. **Maintainability**: Easier to find and modify specific functionality
4. **Scalability**: New applications can easily import common components
5. **Clean Separation**: Forecast system is isolated from store simulation

## Running Applications

### Sales Forecasting

```bash
python forecast/sales_forecast_app.py
```

### Store Simulation

```bash
python store.py
```

### Discrete Event Simulation

```bash
python sim.py <arrival_rates> <multiplier> <time_unit>
```

## File Dependencies

- **Forecast system**: Self-contained in `forecast/` folder, imports from `common/`
- **Store simulation**: Imports UI components from `common/`
- **Assets**: Shared data files accessible from both applications

The refactoring maintains full backward compatibility while providing a much cleaner and more maintainable codebase structure.
