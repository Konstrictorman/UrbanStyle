"""
Gymnasium environment for dynamic pricing RL
Implements a complete RL environment for price optimization
"""

import gymnasium as gym
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from gymnasium import spaces
from customer_simulation import CustomerSimulation
from data_processor import SalesDataProcessor
import os


class DynamicPricingEnv(gym.Env):
    """
    Dynamic pricing environment for reinforcement learning
    
    State: (current_price, segment_id, recent_sales, week, reference_price)
    Actions: Price adjustments (-10% to +10% in 2% increments)
    Reward: Revenue = (price - cost) * demand
    """
    
    def __init__(
        self,
        segment_name: str = "Electronics",
        csv_path: str = None,
        initial_price_multiplier: float = 1.0,
    ):
        """
        Initialize the dynamic pricing environment
        
        Args:
            segment_name: Name of the product segment to optimize
            csv_path: Path to the CSV data file
        """
        super().__init__()
        
        # Environment parameters
        self.segment_name = segment_name
        self.max_weeks = 54  # From CSV data
        self.current_week = 0
        
        # Load and process data
        if csv_path is None:
            # Default path
            assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
            csv_path = os.path.join(assets_dir, "Kagle.csv")
        
        self.data_processor = SalesDataProcessor(csv_path)
        self.data_processor.load_data()
        self.segment_info = self.data_processor.get_segment_info(segment_name)
        
        if not self.segment_info:
            raise ValueError(f"Segment '{segment_name}' not found in data")
        
        # Initialize customer simulation
        self.customer_sim = CustomerSimulation(self.segment_info)
        
        # State variables
        self.initial_price_multiplier = max(0.1, initial_price_multiplier)
        self.current_price = self.segment_info['reference_price'] * self.initial_price_multiplier
        min_price = self.segment_info['cost_per_unit'] * 1.01
        self.current_price = max(self.current_price, min_price)
        self.reference_price = self.segment_info['reference_price']
        self.cost_per_unit = self.segment_info['cost_per_unit']
        self.recent_sales = [0.0] * 4  # Last 4 weeks of sales
        
        # Action space: 11 discrete actions (price adjustments)
        # Actions: [-10%, -8%, -6%, -4%, -2%, 0%, +2%, +4%, +6%, +8%, +10%]
        self.action_space = spaces.Discrete(11)
        
        # State space: (price, segment, recent_sales, week, reference_price)
        self.observation_space = spaces.Dict({
            'current_price': spaces.Box(low=0.0, high=1000.0, shape=(1,), dtype=np.float32),
            'segment_id': spaces.Box(low=0, high=2, shape=(1,), dtype=np.int32),
            'recent_sales': spaces.Box(low=0.0, high=1000.0, shape=(4,), dtype=np.float32),
            'week': spaces.Box(low=0, high=54, shape=(1,), dtype=np.int32),
            'reference_price': spaces.Box(low=0.0, high=1000.0, shape=(1,), dtype=np.float32)
        })
        
        # Action mapping
        self.action_to_price_change = {
            0: -0.10,   # -10%
            1: -0.08,   # -8%
            2: -0.06,   # -6%
            3: -0.04,   # -4%
            4: -0.02,   # -2%
            5: 0.00,    # 0%
            6: 0.02,    # +2%
            7: 0.04,    # +4%
            8: 0.06,    # +6%
            9: 0.08,    # +8%
            10: 0.10    # +10%
        }
        
        # Tracking variables
        self.episode_revenue = 0.0
        self.episode_rewards = []
        self.price_history = []
        self.demand_history = []
        self.revenue_history = []
        
        print(f"Dynamic Pricing Environment initialized for {segment_name}")
        print(f"Reference price: ${self.reference_price:.2f}")
        print(f"Cost per unit: ${self.cost_per_unit:.2f}")
        print(f"Max weeks per episode: {self.max_weeks}")
    
    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Convert internal state to observation format"""
        return {
            'current_price': np.array([self.current_price], dtype=np.float32),
            'segment_id': np.array([self.segment_info['segment_id']], dtype=np.int32),
            'recent_sales': np.array(self.recent_sales, dtype=np.float32),
            'week': np.array([self.current_week], dtype=np.int32),
            'reference_price': np.array([self.reference_price], dtype=np.float32)
        }
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information for debugging"""
        return {
            'week': self.current_week,
            'current_price': self.current_price,
            'reference_price': self.reference_price,
            'cost_per_unit': self.cost_per_unit,
            'segment_name': self.segment_name,
            'episode_revenue': self.episode_revenue
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """
        Reset the environment to start a new episode
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)
            
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        
        # Reset episode variables
        self.current_week = 0
        self.current_price = self.reference_price * self.initial_price_multiplier
        min_price = self.cost_per_unit * 1.01
        self.current_price = max(self.current_price, min_price)
        self.recent_sales = [0.0] * 4
        self.episode_revenue = 0.0
        self.episode_rewards = []
        self.price_history = []
        self.demand_history = []
        self.revenue_history = []
        
        # Reset customer simulation
        self.customer_sim.reset_history()
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one timestep within the environment
        
        Args:
            action: The action to take (0-10 for price adjustments)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Ensure action is a plain integer (SB3 may return numpy scalar)
        if isinstance(action, np.ndarray):
            if action.size != 1:
                raise ValueError(f"Invalid action shape: {action}")
            action = int(action.item())
        else:
            action = int(action)
        
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        
        # Apply price change
        price_change = self.action_to_price_change[action]
        new_price = self.current_price * (1 + price_change)
        
        # Ensure price stays within reasonable bounds
        min_price = self.cost_per_unit * 1.01  # At least 1% markup
        max_price = self.reference_price * 3.0  # Max 300% of reference
        new_price = np.clip(new_price, min_price, max_price)
        
        # Update current price
        self.current_price = new_price
        
        # Calculate demand and revenue
        demand = self.customer_sim.calculate_demand(self.current_price)
        revenue = (self.current_price - self.cost_per_unit) * demand
        
        # Calculate reward (normalized revenue)
        max_possible_revenue = (self.reference_price * 2.0 - self.cost_per_unit) * self.segment_info['base_demand']
        reward = revenue / max_possible_revenue if max_possible_revenue > 0 else 0
        
        # Update tracking variables
        self.episode_revenue += revenue
        self.episode_rewards.append(reward)
        self.price_history.append(self.current_price)
        self.demand_history.append(demand)
        self.revenue_history.append(revenue)
        
        # Update recent sales (rolling window)
        self.recent_sales.pop(0)
        self.recent_sales.append(demand)
        
        # Advance week
        self.current_week += 1
        
        # Check termination conditions
        terminated = self.current_week >= self.max_weeks
        truncated = False  # No truncation in this environment
        
        # Get observation and info
        observation = self._get_obs()
        info = self._get_info()
        info.update({
            'demand': demand,
            'revenue': revenue,
            'price_change': price_change,
            'new_price': new_price
        })
        
        return observation, reward, terminated, truncated, info
    
    def render(self, mode: str = "human") -> Optional[str]:
        """
        Render the environment
        
        Args:
            mode: Render mode ('human' for console output)
            
        Returns:
            Optional string representation
        """
        if mode == "human":
            print(f"Week {self.current_week}/{self.max_weeks}")
            print(f"Current Price: ${self.current_price:.2f}")
            print(f"Reference Price: ${self.reference_price:.2f}")
            print(f"Recent Sales: {self.recent_sales}")
            print(f"Episode Revenue: ${self.episode_revenue:.2f}")
            print("-" * 40)
    
    def get_episode_stats(self) -> Dict[str, Any]:
        """
        Get statistics from the current episode
        
        Returns:
            Dictionary with episode statistics
        """
        if not self.episode_rewards:
            return {}
        
        return {
            'total_revenue': self.episode_revenue,
            'avg_reward': np.mean(self.episode_rewards),
            'total_reward': np.sum(self.episode_rewards),
            'avg_price': np.mean(self.price_history) if self.price_history else 0,
            'avg_demand': np.mean(self.demand_history) if self.demand_history else 0,
            'price_volatility': np.std(self.price_history) if self.price_history else 0,
            'weeks_completed': self.current_week,
            'initial_price_multiplier': self.initial_price_multiplier
        }
    
    def get_static_pricing_baseline(self) -> float:
        """
        Calculate baseline revenue using static pricing
        
        Returns:
            Total revenue from static pricing
        """
        static_price = self.reference_price
        total_revenue = 0.0
        
        for week in range(self.max_weeks):
            demand = self.customer_sim.calculate_demand(static_price)
            revenue = (static_price - self.cost_per_unit) * demand
            total_revenue += revenue
        
        return total_revenue
    
    def close(self):
        """Close the environment"""
        pass

    def set_initial_price_multiplier(self, multiplier: float):
        """
        Update the initial price multiplier used on reset.
        """
        self.initial_price_multiplier = max(0.1, float(multiplier))


def main():
    """Test the dynamic pricing environment"""
    try:
        # Create environment
        env = DynamicPricingEnv(segment_name="Electronics")
        
        # Test episode
        obs, info = env.reset(seed=42)
        print("Environment reset successfully")
        print(f"Initial observation: {obs}")
        
        # Run a few steps
        for step in range(5):
            action = env.action_space.sample()  # Random action
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step {step + 1}: Action {action}, Reward {reward:.4f}")
            
            if terminated:
                break
        
        # Get episode stats
        stats = env.get_episode_stats()
        print(f"\nEpisode stats: {stats}")
        
        # Get static pricing baseline
        baseline_revenue = env.get_static_pricing_baseline()
        print(f"Static pricing baseline revenue: ${baseline_revenue:.2f}")
        
        env.close()
        
    except Exception as e:
        print(f"Error testing environment: {e}")


if __name__ == "__main__":
    main()
