# Project Structure Diagram

```
UrbanStyle/
├── 📁 forecast/                    # Sales Forecasting System
│   ├── sales_forecast_app.py      # Main Pygame GUI application
│   ├── sales_data_processor.py    # Data preprocessing & features
│   ├── sales_lstm_model.py        # LSTM neural network model
│   ├── sales_trainer.py           # Training & evaluation logic
│   ├── forecast_plotter.py         # Single-category plotting
│   ├── multi_category_trainer.py   # Multi-category training
│   ├── multi_category_plotter.py  # Multi-category plots (separate)
│   ├── single_plot_multi_category_plotter.py  # Multi-category plots (unified)
│   └── checkbox.py                # Checkbox UI component
│
├── 📁 common/                      # Shared UI Components
│   ├── button.py                  # Button widget
│   ├── slider.py                  # Slider widget
│   ├── inputfield.py              # Text input field
│   └── progressbar.py             # Progress bar widget
│
├── 📁 assets/                      # Data Files
│   ├── Kagle_.csv                 # Sales transaction data
│   ├── footprints.svg              # Store layout graphics
│   └── ...                        # Other assets
│
├── 📁 Root Directory               # Core Applications
│   ├── store.py                   # Q-learning store simulation
│   ├── sim.py                     # Discrete event simulation
│   ├── data.py                    # Simulation data structures
│   ├── forecast.py                # Legacy sine wave forecasting
│   ├── tracker.py                 # Performance tracking
│   ├── requirements.txt           # Python dependencies
│   └── README_Refactoring.md      # This documentation
│
└── 📁 Other Files
    ├── README.md                   # Main project documentation
    ├── README_SalesForecast.md     # Sales forecast documentation
    └── analyze_and_extend_csv.py   # Data analysis utilities
```

## Import Flow

```
forecast/sales_forecast_app.py
    ↓ imports from
common/ (slider, button)
    ↓ and from
forecast/ (trainer, plotter)

store.py
    ↓ imports from
common/ (button, progressbar, inputfield)
```

## Application Entry Points

- **Sales Forecasting**: `python forecast/sales_forecast_app.py`
- **Store Simulation**: `python store.py`
- **Discrete Event Sim**: `python sim.py <args>`
