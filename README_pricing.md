# Dynamic Pricing Reinforcement Learning Environment

The UrbanStyle dynamic pricing platform integrates empirical sales data, stochastic customer simulations, and deep reinforcement learning to explore optimal pricing policies. This document reports the current system design, theoretical underpinnings, and user-facing workflow with an academic focus on reproducibility and transparency.

## 1. System Overview

The platform ingests transaction-level data (`assets/Kagle.csv`), constructs weekly demand signals per product segment, and exposes a Gymnasium-compliant environment. Stable-Baselines3 agents (primarily Proximal Policy Optimization, PPO) interact with the environment to learn revenue-maximizing pricing strategies. A Pygame front-end (`pricing_app.py`) orchestrates experiments, allowing practitioners to configure hyperparameters, launch training, and visualize historical versus policy-driven revenues.

### 1.1 Functional Highlights

- **Data realism**: Feature engineering preserves segment-specific reference prices, costs (85 % of price), and utility margins (15 %).
- **Customer modelling**: A logistic demand curve parameterized by segment-level price sensitivity captures diminishing demand under price increases.
- **RL integration**: PPO or DQN agents operate on a state vector containing current price, segment identifier, recent sales, calendar week, and reference price.
- **Visualization**: The Pygame interface streams training rewards, revenue comparisons, and evaluation diagnostics in real time.
- **User controls**: Two sliders govern initial price multipliers (used to seed the environment) and the total number of PPO timesteps (proxy for sample complexity).

## 2. Architectural Components

| Module | Role |
|--------|------|
| `data_processor.py` | Parses CSV data, computes summary statistics, derives weekly aggregates, and exposes segment descriptors (reference price, base demand, price sensitivity). |
| `customer_simulation.py` | Implements the logistic demand model and auxiliary routines such as cost-aware optimal-price search. |
| `dynamic_pricing_env.py` | Gymnasium `Env` with discrete price-adjustment actions (−10 %…+10 %). Observations combine continuous price variables with categorical/temporal context. Rewards normalize weekly revenue. |
| `pricing_agent.py` | Wrapper over Stable-Baselines3 (PPO/DQN). Handles environment vectorization, hyperparameter management, checkpointing, and evaluation. |
| `pricing_app.py` | Pygame application that coordinates training threads, processes slider inputs, runs evaluations post-training, and renders diagnostic plots. |

## 3. Pricing Application (`pricing_app.py`)

The application constitutes the interactive layer between end-users and the RL back end. Its design balances asynchronous training with deterministic visualization to avoid contention between Pygame’s event loop and PyTorch operations.

### 3.1 Workflow

1. **Parameter selection**: Users adjust (i) the *Initial Price Multiplier* slider, scaling each segment’s reference price before training, and (ii) the *Training Timesteps* slider, defining the PPO `total_timesteps`. Each environment episode spans 54 weeks, so the effective number of episodes ≈ `timesteps / 54`.
2. **Training execution**: Pressing “Start Training” spawns a background thread that:
   - Instantiates fresh `PricingAgent` and `DynamicPricingEnv` objects seeded with the slider values.
   - Computes a static-pricing baseline for comparability.
   - Invokes `agent.train(...)` with Stable-Baselines3, relaying progress to the GUI via callbacks (episode rewards, percentage completion).
3. **Post-training evaluation**: Upon completion, the thread schedules an evaluation task on the main thread. This task runs three rollout episodes, captures weekly dynamic revenues, recomputes the static baseline, and updates the Matplotlib plots. Evaluation can also be triggered on demand via the “Evaluate” button.
4. **Visualization**: The upper plot presents the training reward trajectory, while the lower plot juxtaposes static versus RL revenues for the most recent evaluation. When no data is available, informative prompts guide the user to train first.

### 3.2 Safety and Responsiveness

- Buttons automatically enable/disable based on training/evaluation status, preventing concurrent SB3 invocations.
- Evaluation runs on the main thread to avoid macOS semaphore leaks observed when heavy Pandas/Gym workloads execute inside training threads.
- The reset button reinitializes the environment with the current initial-price multiplier, clearing progress indicators to encourage reproducible experiments.

## 4. Mathematical Model

The demand probability for a given week is modelled as:

\[
P(\text{purchase}) = \frac{1}{1 + \exp\left(-\alpha \left(p_{\text{ref}} - p_{\text{cur}}\right)\right)}
\]

where:

- \( p_{\text{ref}} \) is the reference (historical) price,
- \( p_{\text{cur}} \) is the RL-selected price,
- \( \alpha \) is the segment-specific sensitivity parameter.

Demand realizations propagate through `customer_simulation.py`, which returns expected quantities and revenue. The RL reward is the normalized revenue relative to an upper bound defined by \( 2p_{\text{ref}} \) minus cost.

## 5. Usage Instructions

1. **Install dependencies**
   ```bash
   pip install -r requirements_pricing.txt
   ```
2. **Launch the Pygame app**
   ```bash
   python pricing_app.py
   ```
3. **Configure sliders**
   - *Initial Price Multiplier*: scalar ∈ [0.5, 2.0]; also applied instantaneously to the live environment.
   - *Training Timesteps*: integer ∈ [1 000, 50 000]; directly passed to PPO `total_timesteps`.
4. **Start training** with the button; observe live plots.
5. **After training**, the system will automatically evaluate and plot results. Use the “Evaluate” button for additional rollouts with updated multipliers.

### Programmatic Training (for reproducibility)

```python
from pricing_agent import PricingAgent

agent = PricingAgent(algorithm="PPO", segment_name="Electronics")
agent.create_environment()
agent.create_model()
training_stats = agent.train(total_timesteps=10_000)

mean_reward, std_reward = agent.evaluate(n_eval_episodes=5)
episode_stats = agent.run_episode()
```

## 6. Expected Behaviour

- **Revenue uplift**: Empirically 10–20 % over static pricing for data in `Kagle.csv`.
- **Learning dynamics**: PPO reward curves typically stabilize within 10 000–20 000 timesteps when the environment provides informative gradients (non-zero demand variance).
- **Interpretability**: The evaluation console printout reports static revenue, RL revenue, absolute improvement, and average price during the last rollout to aid managerial interpretation.

## 7. Visualization Details

- **Training plot**: Streams per-episode rewards; absence of data prompts the user to initiate training.
- **Revenue plot**: Overlays static (dashed red) and RL (solid green) weekly revenues. Missing data prompts indicate the need to train/evaluate.
- **Status indicator**: Reports “Training,” “Evaluating,” or “Idle,” aligned with button states.
- **Progress bar**: Displays the PPO callback-derived completion percentage.

## 8. File Structure

```
utility-rl/
├── data_processor.py
├── customer_simulation.py
├── dynamic_pricing_env.py
├── pricing_agent.py
├── pricing_app.py
├── README_pricing.md
├── requirements_pricing.txt
└── test_pricing_system.py
```

## 9. Testing

Execute `python test_pricing_system.py` to exercise data loading, environment transitions, agent training, and integration hooks. Additional manual testing via the GUI is recommended after major UI or threading changes.

## 10. Future Directions

- Multi-product joint optimization and cross-elasticities.
- Competitive-response modelling through multi-agent RL.
- Seasonal and promotional effects via exogenous calendar features.
- Persistent model checkpoint management within the UI for longitudinal experiments.

## 11. Troubleshooting

| Symptom | Resolution |
|---------|------------|
| `Kagle.csv` not found | Confirm the file resides under `assets/` and the path in `pricing_app.py` is correct. |
| MacOS “trace trap” during training | Ensure training completes before closing the window; avoid manual termination while PPO runs. |
| Blank revenue plot | Train the agent or press “Evaluate” after training to generate revenue series. |
| Sliders have no effect | The initial-price slider updates immediately; the timesteps slider is read at the next training launch. Verify the console logs show the expected values. |

---

This documentation reflects the system state after the removal of manual simulation buttons, the introduction of automatic post-training evaluation, and the enforced single-threaded evaluation workflow for stability. Continuous feedback is encouraged to further refine both the experimental pipeline and the academic reproducibility of results.
