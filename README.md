# Demo Videos

Please see each of the .mov files for each of the tasks on the demos folder 👾

- 🎥 `UrbanStyle_storepath.mov`
- 🎥 `UrbanStyle_forecasting.mov`
- 🎥 `UrbanStyle_pricing.mov`
- 🎥 `UrbanStyle_fashion-gan.mov`

# Project Structure

```
UrbanStyle/
├── assets/                         # Shared datasets & media
│   ├── Kagle_.csv                  # Enriched retail transactions
│   ├── retail_sales_dataset.csv    # Supporting datasets
│   └── footprints*.svg/png         # Storepath visuals
│
├── common/                         # Reusable UI widgets
│   ├── button.py                   # Styled button component
│   ├── slider.py                   # Generic slider + panel helpers
│   ├── inputfield.py               # Text input widget
│   └── progressbar.py              # Lightweight progress bar
│
├── forecast/                       # Sales-forecasting application
│   ├── sales_forecast_app.py       # Pygame UI + workflow coordinator
│   ├── sales_data_processor.py     # Category-aware preprocessing
│   ├── sales_trainer.py            # Async LSTM training orchestration
│   ├── sales_lstm_model.py         # Forecasting network definition
│   ├── forecast_plotter.py         # Active single-plot renderer
│   ├── multi_category_plotter.py   # Legacy plotter variants
│   ├── multi_category_trainer.py   # Legacy trainer (retained for reference)
│   └── checkbox.py                 # Checkbox control used by UI
│
├── utility-rl/                     # Dynamic pricing RL lab
│   ├── pricing_app.py              # Interactive PPO training UI
│   ├── pricing_agent.py            # Stable-Baselines3 wrapper
│   ├── dynamic_pricing_env.py      # Gymnasium environment
│   ├── data_processor.py           # Segment analytics for RL
│   ├── customer_simulation.py      # Logistic demand model
│   ├── requirements_pricing.txt    # RL-specific dependencies
│   └── test_pricing_system.py      # Integration tests
│
├── fashion-gan/                    # Fashion image generation app
│   ├── gan_app.py                  # Pygame DCGAN trainer/viewer
│   ├── train_gan.py                # Scriptable training entry point
│   ├── samples/, checkpoints/, out/ # Generated samples & weights
│   └── requirements.txt            # GAN dependencies
│
├── storepath/                      # Store navigation & Q-learning tools
│   ├── store.py                    # Main storepath UI and simulation
│   ├── storeQLearning.py           # Algorithmic training helpers
│   └── tracker.py                  # Path-tracking utilities
│
├── root files
│   ├── data.py / sim.py            # Discrete-event simulation artifacts
│   ├── requirements.txt            # Base dependencies
│   ├── README_*.md                 # Per-subsystem documentation
│   ├── *.mov                       # Demo recordings
│   └── pricing_model_electronics_ppo.zip  # Exported RL policy
└── README.md (this file)           # Top-level guide
```

## Application Entry Points

- **Store Simulation**: `python storepath/store.py`
- **Sales Forecasting**: `python forecast/sales_forecast_app.py`
- **Utility RL**: `python utility-rl/pricing_app.py`
- **Fashion GAN**: `python fashion-gan/gan_app.py`
