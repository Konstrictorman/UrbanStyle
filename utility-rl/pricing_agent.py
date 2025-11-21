"""
RL agent wrapper for stable-baselines3 integration
Handles training, evaluation, and model management for dynamic pricing
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
import os
import pickle
from datetime import datetime

# Stable-baselines3 imports
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import FlattenObservation

# Custom imports
from dynamic_pricing_env import DynamicPricingEnv


class PricingTrainingCallback(BaseCallback):
    """
    Custom callback for tracking training progress
    """
    
    def __init__(
        self,
        check_freq: int,
        save_path: str,
        total_timesteps: int,
        progress_callback: Optional[callable] = None,
        verbose: int = 1,
    ):
        """
        Initialize callback
        
        Args:
            check_freq: How often to check and save (in episodes)
            save_path: Path to save the model
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_mean_reward = -np.inf
        self.total_timesteps = max(1, total_timesteps)
        self.progress_callback = progress_callback
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_losses = []
        
    def _on_step(self) -> bool:
        """
        Called at each step during training
        
        Returns:
            bool: Whether to continue training
        """
        # Log episode info if available
        if 'episode' in self.locals['infos'][0]:
            episode_info = self.locals['infos'][0]['episode']
            self.episode_rewards.append(episode_info['r'])
            self.episode_lengths.append(episode_info['l'])
        
        # Save model periodically
        if self.n_calls % self.check_freq == 0:
            self.model.save(self.save_path)
            if self.verbose > 0:
                print(f"Model saved at step {self.n_calls}")
        
        if self.progress_callback:
            progress = min(1.0, self.n_calls / self.total_timesteps)
            try:
                self.progress_callback(progress, self.locals)
            except Exception as exc:
                if self.verbose > 0:
                    print(f"Progress callback error: {exc}")
        
        return True
    
    def get_training_stats(self) -> Dict[str, Any]:
        """
        Get training statistics
        
        Returns:
            Dictionary with training statistics
        """
        if not self.episode_rewards:
            return {}
        
        return {
            'total_episodes': len(self.episode_rewards),
            'mean_reward': np.mean(self.episode_rewards),
            'std_reward': np.std(self.episode_rewards),
            'mean_episode_length': np.mean(self.episode_lengths),
            'best_reward': np.max(self.episode_rewards),
            'worst_reward': np.min(self.episode_rewards)
        }


class PricingAgent:
    """
    Wrapper class for RL agent training and evaluation
    """
    
    def __init__(self, algorithm: str = "PPO", segment_name: str = "Electronics", 
                 model_save_path: str = None):
        """
        Initialize pricing agent
        
        Args:
            algorithm: RL algorithm to use ('PPO' or 'DQN')
            segment_name: Product segment to optimize
            model_save_path: Path to save/load model
        """
        self.algorithm = algorithm.upper()
        self.segment_name = segment_name
        self.model_save_path = model_save_path or f"pricing_model_{segment_name.lower()}_{algorithm.lower()}"
        
        # Initialize environment
        self.env = None
        self.model = None
        self.training_callback = None
        
        # Training metrics
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'evaluation_rewards': [],
            'training_losses': []
        }
        
        print(f"Pricing agent initialized with {algorithm} for {segment_name}")
    
    def create_environment(self, n_envs: int = 1, env_kwargs: Optional[Dict[str, Any]] = None) -> Any:
        """
        Create vectorized environment for training
        
        Args:
            n_envs: Number of parallel environments
            
        Returns:
            Vectorized environment
        """
        env_kwargs = env_kwargs or {}
        env_kwargs.setdefault('segment_name', self.segment_name)
        
        def make_env():
            env = DynamicPricingEnv(**env_kwargs)
            env = FlattenObservation(env)
            env = Monitor(env)  # Monitor for episode statistics
            return env
        
        if n_envs == 1:
            self.env = make_env()
        else:
            self.env = make_vec_env(make_env, n_envs=n_envs)
        
        return self.env
    
    def create_model(self, **kwargs) -> Any:
        """
        Create RL model with specified algorithm
        
        Args:
            **kwargs: Additional arguments for model creation
            
        Returns:
            Trained model
        """
        if self.env is None:
            self.create_environment()
        
        # Default hyperparameters
        default_params = {
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'verbose': 1
        }
        
        # Update with user-provided parameters
        default_params.update(kwargs)
        
        if self.algorithm == "PPO":
            self.model = PPO("MlpPolicy", self.env, **default_params)
        elif self.algorithm == "DQN":
            # DQN-specific parameters
            dqn_params = {
                'learning_rate': 1e-4,
                'buffer_size': 10000,
                'learning_starts': 1000,
                'batch_size': 32,
                'tau': 1.0,
                'gamma': 0.99,
                'train_freq': 4,
                'gradient_steps': 1,
                'target_update_interval': 1000,
                'exploration_fraction': 0.1,
                'exploration_initial_eps': 1.0,
                'exploration_final_eps': 0.05,
                'max_grad_norm': 10,
                'verbose': 1
            }
            dqn_params.update(kwargs)
            self.model = DQN("MlpPolicy", self.env, **dqn_params)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        
        print(f"Model created with {self.algorithm} algorithm")
        return self.model
    
    def train(self, total_timesteps: int = 100000, 
              callback_freq: int = 10000,
              progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Train the RL agent
        
        Args:
            total_timesteps: Total number of training timesteps
            callback_freq: Frequency for callbacks
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with training results
        """
        if self.model is None:
            self.create_model()
        
        # Create training callback
        self.training_callback = PricingTrainingCallback(
            check_freq=callback_freq,
            save_path=self.model_save_path,
            total_timesteps=total_timesteps,
            progress_callback=progress_callback,
            verbose=1
        )
        
        print(f"Starting training for {total_timesteps} timesteps...")
        
        # Train the model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=self.training_callback,
            progress_bar=True
        )
        
        # Save final model
        self.model.save(self.model_save_path)
        print(f"Training completed. Model saved to {self.model_save_path}")
        
        # Get training statistics
        training_stats = self.training_callback.get_training_stats()
        training_stats['total_timesteps'] = total_timesteps
        training_stats['algorithm'] = self.algorithm
        training_stats['segment_name'] = self.segment_name
        
        return training_stats
    
    def evaluate(self, n_eval_episodes: int = 10, 
                 deterministic: bool = True,
                 env_kwargs: Optional[Dict[str, Any]] = None) -> Tuple[float, float]:
        """
        Evaluate the trained agent
        
        Args:
            n_eval_episodes: Number of episodes to evaluate
            deterministic: Whether to use deterministic actions
            
        Returns:
            Tuple of (mean_reward, std_reward)
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Create evaluation environment
        env_kwargs = env_kwargs or {}
        env_kwargs.setdefault('segment_name', self.segment_name)
        eval_env = DynamicPricingEnv(**env_kwargs)
        eval_env = FlattenObservation(eval_env)
        eval_env = Monitor(eval_env)
        
        # Evaluate the policy
        mean_reward, std_reward = evaluate_policy(
            self.model, eval_env, n_eval_episodes=n_eval_episodes,
            deterministic=deterministic, return_episode_rewards=True
        )
        
        print(f"Evaluation results:")
        print(f"  Mean reward: {mean_reward:.4f} ± {std_reward:.4f}")
        
        eval_env.close()
        return mean_reward, std_reward
    
    def predict(self, observation: Dict, deterministic: bool = True) -> Tuple[int, np.ndarray]:
        """
        Predict action for given observation
        
        Args:
            observation: Current state observation
            deterministic: Whether to use deterministic prediction
            
        Returns:
            Tuple of (action, state)
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        action, state = self.model.predict(observation, deterministic=deterministic)
        if isinstance(action, np.ndarray):
            if action.size != 1:
                raise ValueError(f"Predicted action has unexpected shape: {action}")
            action = int(action.item())
        else:
            action = int(action)
        return action, state
    
    def run_episode(self, render: bool = False, env_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run a single episode with the trained agent
        
        Args:
            render: Whether to render the environment
            
        Returns:
            Dictionary with episode results
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Create environment
        env_kwargs = env_kwargs or {}
        env_kwargs.setdefault('segment_name', self.segment_name)
        base_env = DynamicPricingEnv(**env_kwargs)
        env = FlattenObservation(base_env)
        obs, info = env.reset()
        
        episode_rewards = []
        episode_actions = []
        episode_prices = []
        episode_demands = []
        episode_revenues = []
        
        done = False
        while not done:
            action, _ = self.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_rewards.append(reward)
            episode_actions.append(action)
            episode_prices.append(info['current_price'])
            episode_demands.append(info['demand'])
            episode_revenues.append(info.get('revenue', 0.0))
            
            if render:
                env.render()
            
            done = terminated or truncated
        
        # Calculate episode statistics
        episode_stats = {
            'total_reward': np.sum(episode_rewards),
            'mean_reward': np.mean(episode_rewards),
            'episode_length': len(episode_rewards),
            'final_price': episode_prices[-1] if episode_prices else 0,
            'avg_price': np.mean(episode_prices),
            'avg_demand': np.mean(episode_demands),
            'price_volatility': np.std(episode_prices),
            'actions': episode_actions,
            'prices': episode_prices,
            'demands': episode_demands,
            'revenues': episode_revenues,
            'segment_name': self.segment_name,
            'initial_price_multiplier': base_env.initial_price_multiplier
        }
        
        env.close()
        return episode_stats
    
    def save_model(self, path: str = None):
        """
        Save the trained model
        
        Args:
            path: Path to save the model (optional)
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        save_path = path or self.model_save_path
        self.model.save(save_path)
        print(f"Model saved to {save_path}")
    
    def load_model(self, path: str = None):
        """
        Load a trained model
        
        Args:
            path: Path to load the model from (optional)
        """
        load_path = path or self.model_save_path
        
        if self.algorithm == "PPO":
            self.model = PPO.load(load_path)
        elif self.algorithm == "DQN":
            self.model = DQN.load(load_path)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        
        print(f"Model loaded from {load_path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {'status': 'No model loaded'}
        
        return {
            'algorithm': self.algorithm,
            'segment_name': self.segment_name,
            'model_path': self.model_save_path,
            'policy': str(self.model.policy),
            'learning_rate': getattr(self.model, 'learning_rate', 'N/A'),
            'gamma': getattr(self.model, 'gamma', 'N/A')
        }


def main():
    """Test the pricing agent"""
    try:
        # Create agent
        agent = PricingAgent(algorithm="PPO", segment_name="Electronics")
        
        # Create environment and model
        agent.create_environment()
        agent.create_model()
        
        # Train for a short period
        training_stats = agent.train(total_timesteps=10000)
        print(f"Training stats: {training_stats}")
        
        # Evaluate
        mean_reward, std_reward = agent.evaluate(n_eval_episodes=5)
        
        # Run a sample episode
        episode_stats = agent.run_episode(render=True)
        print(f"Episode stats: {episode_stats}")
        
        # Get model info
        model_info = agent.get_model_info()
        print(f"Model info: {model_info}")
        
    except Exception as e:
        print(f"Error testing agent: {e}")


if __name__ == "__main__":
    main()
