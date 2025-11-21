"""
Main Pygame application for dynamic pricing RL environment
Provides interactive visualization and training interface
"""

import pygame
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import threading
import traceback
from typing import Dict, List, Optional, Any
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import custom modules
from dynamic_pricing_env import DynamicPricingEnv
from pricing_agent import PricingAgent
from data_processor import SalesDataProcessor
from customer_simulation import CustomerSimulation

# Import UI components from common folder
from common.button import Button
from common.slider import Slider

# Pygame constants
WINDOW_WIDTH = 1420
WINDOW_HEIGHT = 800
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
DARK_GRAY = (64, 64, 64)
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 128)




class ProgressBar:
    """Progress bar component for Pygame"""
    
    def __init__(self, x: int, y: int, width: int, height: int, 
                 max_value: float = 100.0):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.max_value = max_value
        self.current_value = 0.0
        
    def set_value(self, value: float):
        """Set current progress value"""
        self.current_value = max(0, min(value, self.max_value))
        
    def draw(self, screen: pygame.Surface, font: pygame.font.Font):
        """Draw the progress bar"""
        # Draw background
        pygame.draw.rect(screen, LIGHT_GRAY, (self.x, self.y, self.width, self.height))
        pygame.draw.rect(screen, DARK_GRAY, (self.x, self.y, self.width, self.height), 2)
        
        # Draw progress
        progress_width = int((self.current_value / self.max_value) * self.width)
        if progress_width > 0:
            pygame.draw.rect(screen, GREEN, (self.x, self.y, progress_width, self.height))
        
        # Draw percentage text
        percentage = (self.current_value / self.max_value) * 100
        text = font.render(f"{percentage:.1f}%", True, BLACK)
        text_rect = text.get_rect(center=(self.x + self.width // 2, self.y + self.height // 2))
        screen.blit(text, text_rect)


class PricingApp:
    """Main application class for dynamic pricing RL"""
    
    def __init__(self):
        """Initialize the pricing application"""
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Dynamic Pricing RL Environment")
        self.clock = pygame.time.Clock()
        
        # Fonts (reduced by ~20%)
        self.font = pygame.font.Font(None, 20)
        self.title_font = pygame.font.Font(None, 26)
        self.small_font = pygame.font.Font(None, 15)
        
        # Application state
        self.running = True
        self.training = False
        self.agent = None
        self.env = None
        self.data_processor = None
        
        # Training variables
        self.training_thread = None
        self.training_progress = 0.0
        self.training_episodes = 0
        self.training_rewards = []
        self.static_revenue = 0.0
        self.dynamic_revenue = 0.0
        
        # Simulation data arrays
        self.weekly_static_revenues = []
        self.weekly_dynamic_revenues = []
        self.weekly_prices = []
        self.weekly_demands = []
        self.pending_eval_kwargs = None
        self.evaluating = False
        
        # UI Components
        self.setup_ui()
        self.update_button_states()
        
        # RL configuration
        self.segment_name = "Electronics"
        self.agent_algorithm = "PPO"
        self.initial_price_multiplier = self.initial_price_slider.get_value()
        self.latest_training_stats: Dict[str, Any] = {}
        self.weekly_static_revenues: List[float] = []
        self.recent_episode_prices: List[float] = []
        self.recent_episode_demands: List[float] = []
        
        # Initialize data
        self.initialize_data()
        
    def setup_ui(self):
        """Setup UI components with proper spacing to avoid overlaps"""
        # Sliders with more vertical spacing
        self.initial_price_slider = Slider(40, 80, 160, 24, 0.5, 2.0, 1.0, "Initial Price Multiplier")
        self.training_episodes_slider = Slider(40, 125, 160, 24, 1000, 50000, 10000, "Training Timesteps", is_integer=True)
        
        # Buttons with more spacing between rows
        self.start_training_button = Button(40, 170, 95, 32, "Start Training", GREEN, on_click=self._start_training_wrapper)
        self.stop_training_button = Button(145, 170, 95, 32, "Stop Training", RED, on_click=self._stop_training_wrapper)
        self.reset_button = Button(40, 220, 95, 32, "Reset", BLUE, on_click=self._reset_wrapper)
        self.evaluate_button = Button(145, 220, 95, 32, "Evaluate", ORANGE, on_click=self._evaluate_wrapper)
        
        # Progress bar with more spacing
        self.progress_bar = ProgressBar(40, 320, 200, 24, 100.0)
        
    def initialize_data(self):
        """Initialize data processor and environment"""
        try:
            # Find CSV file
            assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
            csv_path = os.path.join(assets_dir, "Kagle.csv")
            
            if not os.path.exists(csv_path):
                print(f"CSV file not found at {csv_path}")
                return
            
            # Initialize data processor
            self.data_processor = SalesDataProcessor(csv_path)
            self.data_processor.load_data()
            self.data_processor.process_data()
            
            # Initialize RL components
            self.agent = PricingAgent(algorithm=self.agent_algorithm, segment_name=self.segment_name)
            self.env = DynamicPricingEnv(
                segment_name=self.segment_name,
                initial_price_multiplier=self.initial_price_multiplier
            )
            
            print("Data initialization completed successfully")
            
        except Exception as e:
            print(f"Error initializing data: {e}")
    
    def _start_training_wrapper(self, button):
        """Wrapper for start training button callback"""
        self.start_training()
    
    def _stop_training_wrapper(self, button):
        """Wrapper for stop training button callback"""
        self.stop_training()
    
    def _reset_wrapper(self, button):
        """Wrapper for reset button callback"""
        self.reset_environment()
    
    def _evaluate_wrapper(self, button):
        """Wrapper for evaluate button callback"""
        self.evaluate_agent()
    
    def update_button_states(self):
        """Update enabled/disabled state of control buttons"""
        model_ready = self.agent is not None and self.agent.model is not None
        can_train = not self.training and not self.evaluating
        self.start_training_button.set_enabled(can_train)
        self.stop_training_button.set_enabled(self.training)
        self.reset_button.set_enabled(not self.training and not self.evaluating)
        self.evaluate_button.set_enabled(model_ready and not self.training and not self.evaluating)
    
    def on_initial_price_changed(self, multiplier: float):
        """Apply slider-driven initial price multiplier to environment"""
        self.initial_price_multiplier = multiplier
        if self.env:
            self.env.set_initial_price_multiplier(multiplier)
            if not self.training:
                self.env.reset()
    
    def start_training(self):
        """Start training the RL agent"""
        if self.training:
            return
        
        if self.agent is None:
            self.agent = PricingAgent(algorithm=self.agent_algorithm, segment_name=self.segment_name)
        
        self.training = True
        self.update_button_states()
        self.training_progress = 0.0
        self.training_episodes = 0
        self.training_rewards = []
        
        # Get training parameters from sliders
        initial_price_mult = self.initial_price_slider.get_value()
        total_timesteps = int(self.training_episodes_slider.get_value())
        
        # Start training thread
        self.training_thread = threading.Thread(
            target=self.training_worker,
            args=(initial_price_mult, total_timesteps)
        )
        self.training_thread.daemon = True
        self.training_thread.start()
        
        print("Training started")
    
    def stop_training(self):
        """Stop training the RL agent"""
        self.training = False
        if self.training_thread:
            self.training_thread.join(timeout=1.0)
        self.update_button_states()
        print("Training stopped")
    
    def reset_environment(self):
        """Reset the environment"""
        if self.env:
            self.env.set_initial_price_multiplier(self.initial_price_slider.get_value())
            self.env.reset()
        self.training_progress = 0.0
        self.training_episodes = 0
        self.training_rewards = []
        print("Environment reset")
    
    def evaluate_agent(self):
        """Evaluate the trained agent"""
        if not self.agent or self.agent.model is None:
            print("No trained agent available. Please run training first.")
            return
        
        try:
            env_kwargs = {
                'segment_name': self.segment_name,
                'initial_price_multiplier': self.initial_price_slider.get_value()
            }
            self.run_post_training_evaluation(env_kwargs, episodes=1)
            
            improvement = 0.0
            if self.static_revenue > 0:
                improvement = ((self.dynamic_revenue - self.static_revenue) / self.static_revenue) * 100
            
            print("\n" + "=" * 50)
            print("EVALUATION RESULTS")
            print("=" * 50)
            print(f"Static Pricing Revenue:    ${self.static_revenue:.2f}")
            print(f"RL Pricing Revenue:        ${self.dynamic_revenue:.2f}")
            print(f"Revenue Improvement:       {improvement:.1f}%")
            print(f"Training Episodes Logged:  {self.training_episodes}")
            if self.weekly_dynamic_revenues and self.recent_episode_prices:
                print(f"Last Episode Weeks:        {len(self.weekly_dynamic_revenues)}")
                print(f"Last Episode Avg Price:    ${np.mean(self.recent_episode_prices):.2f}")
            print("=" * 50)
        
        except Exception as e:
            print(f"Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
    
    def training_worker(self, initial_price_mult: float, total_timesteps: int):
        """Worker thread for training"""
        try:
            print("\n==== Training Worker ====")
            print(f"  Segment: {self.segment_name}")
            print(f"  Algorithm: {self.agent_algorithm}")
            print(f"  Initial price multiplier: {initial_price_mult:.3f}")
            print(f"  Timesteps requested: {total_timesteps}")
            
            # Prepare environment configuration
            env_kwargs = {
                'segment_name': self.segment_name,
                'initial_price_multiplier': initial_price_mult
            }
            print(f"  Env kwargs: {env_kwargs}")
            
            # Recreate agent to ensure clean state
            print("  Creating agent/environment/model...")
            self.agent = PricingAgent(algorithm=self.agent_algorithm, segment_name=self.segment_name)
            self.agent.create_environment(env_kwargs=env_kwargs)
            self.agent.create_model()
            
            # Calculate static pricing baseline for comparison
            print("  Calculating static baseline...")
            static_env = DynamicPricingEnv(**env_kwargs)
            self.static_revenue = static_env.get_static_pricing_baseline()
            self.weekly_static_revenues = self._compute_static_weekly_revenue(static_env)
            static_env.close()
            print(f"  Static revenue baseline: {self.static_revenue:.2f}")
            
            def progress_callback(progress: float, callback_locals: Dict[str, Any]):
                self.training_progress = progress * 100.0
                infos = callback_locals.get('infos', [])
                for info in infos:
                    episode_info = info.get('episode')
                    if episode_info:
                        self.training_rewards.append(episode_info['r'])
                        if len(self.training_rewards) > 5000:
                            self.training_rewards = self.training_rewards[-5000:]
                        self.training_episodes = len(self.training_rewards)
            
            callback_freq = max(500, total_timesteps // 10)
            print(f"  Starting SB3 training (callback_freq={callback_freq})...")
            training_stats = self.agent.train(
                total_timesteps=total_timesteps,
                callback_freq=callback_freq,
                progress_callback=progress_callback
            )
            print("  Training stats:")
            print(f"    {training_stats}")
            self.latest_training_stats = training_stats
            self.training_episodes = training_stats.get('total_episodes', self.training_episodes)
            
            # Schedule evaluation to run on the main thread
            self.pending_eval_kwargs = env_kwargs
            print("  Training completed, evaluation queued for main thread.")
            
        except Exception as e:
            print(f"Training error: {e}")
            traceback.print_exc()
        finally:
            self.training = False
            self.update_button_states()
    
    def process_pending_evaluation(self):
        """Run queued evaluation tasks on the main thread"""
        if self.training or self.evaluating or self.pending_eval_kwargs is None:
            return
        
        print("Processing pending evaluation on main thread...")
        self.evaluating = True
        try:
            self.run_post_training_evaluation(self.pending_eval_kwargs, episodes=3)
            print("Pending evaluation completed.")
        except Exception as e:
            print(f"Evaluation error: {e}")
            traceback.print_exc()
        finally:
            self.pending_eval_kwargs = None
            self.evaluating = False
            self.update_button_states()
    
    def run_post_training_evaluation(self, env_kwargs: Dict[str, Any], episodes: int = 3):
        """Evaluate trained agent and capture revenue comparisons"""
        if not self.agent or self.agent.model is None:
            return
        
        dynamic_episode_totals: List[float] = []
        last_episode = None
        
        for _ in range(max(1, episodes)):
            stats = self.agent.run_episode(render=False, env_kwargs=env_kwargs)
            episode_revenue = float(np.sum(stats.get('revenues', [])))
            dynamic_episode_totals.append(episode_revenue)
            last_episode = stats
        
        if last_episode:
            self.weekly_dynamic_revenues = last_episode.get('revenues', [])
            self.weekly_prices = last_episode.get('prices', [])
            self.weekly_demands = last_episode.get('demands', [])
            self.recent_episode_prices = self.weekly_prices[:]
            self.recent_episode_demands = self.weekly_demands[:]
            self.dynamic_revenue = float(np.mean(dynamic_episode_totals))
        
        # Refresh static baseline for comparison
        baseline_env = DynamicPricingEnv(**env_kwargs)
        self.static_revenue = baseline_env.get_static_pricing_baseline()
        self.weekly_static_revenues = self._compute_static_weekly_revenue(baseline_env)
        baseline_env.close()
    
    def _compute_static_weekly_revenue(self, env: DynamicPricingEnv) -> List[float]:
        """Create weekly static revenue series for plotting"""
        revenues = []
        static_price = env.reference_price
        for _ in range(env.max_weeks):
            demand = env.customer_sim.calculate_demand(static_price)
            revenues.append((static_price - env.cost_per_unit) * demand)
        return revenues
    
    
    
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            # Handle slider events
            price_changed = self.initial_price_slider.handle_event(event)
            self.training_episodes_slider.handle_event(event)
            if price_changed:
                self.on_initial_price_changed(self.initial_price_slider.get_value())
            
            # Handle button events using common component method
            mouse_pos = pygame.mouse.get_pos()
            mouse_pressed = pygame.mouse.get_pressed()[0]
            
            self.start_training_button.handle_events(mouse_pos, mouse_pressed)
            self.stop_training_button.handle_events(mouse_pos, mouse_pressed)
            self.reset_button.handle_events(mouse_pos, mouse_pressed)
            self.evaluate_button.handle_events(mouse_pos, mouse_pressed)
    
    def draw_plots(self):
        """Draw training and revenue comparison plots"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 3.5))
        fig.patch.set_edgecolor('black')
        fig.patch.set_linewidth(2)
        
        # Plot 1: Training rewards per episode (live)
        ax1.set_title('Training Reward per Episode', fontsize=9)
        if self.training_rewards:
            ax1.plot(
                range(1, len(self.training_rewards) + 1),
                self.training_rewards,
                color='blue',
                linewidth=1.5,
                label='Episode reward'
            )
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')
            ax1.legend(loc='upper right', prop={'size': 8})
        else:
            ax1.text(0.5, 0.5, 'No training data yet.\nClick "Start Training".',
                     ha='center', va='center', fontsize=8, color='gray', transform=ax1.transAxes)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Static vs dynamic revenue
        ax2.set_title('Revenue Comparison (Most Recent Episode)', fontsize=9)
        if self.weekly_dynamic_revenues and self.weekly_static_revenues:
            weeks = list(range(1, len(self.weekly_dynamic_revenues) + 1))
            static = self.weekly_static_revenues[:len(weeks)]
            ax2.plot(weeks, static, 'r--', linewidth=1.5, label='Static pricing')
            ax2.plot(weeks, self.weekly_dynamic_revenues, 'g-', linewidth=1.8, label='RL pricing')
            ax2.set_xlabel('Week')
            ax2.set_ylabel('Revenue ($)')
            ax2.legend(loc='upper right', prop={'size': 8})
        else:
            ax2.text(0.5, 0.5, 'No revenue data yet.\nTrain the model to generate forecasts.',
                     ha='center', va='center', fontsize=8, color='gray', transform=ax2.transAxes)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.buffer_rgba()
        size = canvas.get_width_height()
        plot_surface = pygame.image.fromstring(bytes(raw_data), size, "RGBA").convert()
        self.screen.blit(plot_surface, (300, 50))
        plt.close(fig)
    
    def draw_ui(self):
        """Draw UI components (reduced spacing)"""
        # Draw title (smaller spacing)
        title_text = self.title_font.render("Dynamic Pricing RL Environment", True, BLACK)
        self.screen.blit(title_text, (40, 15))
        
        # Draw sliders
        self.initial_price_slider.draw(self.screen)
        self.training_episodes_slider.draw(self.screen)
        
        # Draw buttons
        self.start_training_button.draw(self.screen)
        self.stop_training_button.draw(self.screen)
        self.reset_button.draw(self.screen)
        self.evaluate_button.draw(self.screen)
        
        # Draw progress bar
        self.progress_bar.set_value(self.training_progress)
        self.progress_bar.draw(self.screen, self.font)
        
        # Draw training/simulation status with proper spacing
        if self.training:
            status_text = "Training"
            status_color = GREEN
        elif self.evaluating:
            status_text = "Evaluating"
            status_color = ORANGE
        else:
            status_text = "Idle"
            status_color = RED
        status_surface = self.font.render(f"Status: {status_text}", True, status_color)
        self.screen.blit(status_surface, (40, 350))
        
        # Draw episode count
        episode_text = self.font.render(f"Episodes: {self.training_episodes}", True, BLACK)
        self.screen.blit(episode_text, (40, 375))
        
        # Draw revenue comparison
        if self.static_revenue > 0 and self.dynamic_revenue > 0:
            improvement = ((self.dynamic_revenue - self.static_revenue) / self.static_revenue) * 100
            revenue_text = self.font.render(
                f"Static ${self.static_revenue:,.0f} | RL ${self.dynamic_revenue:,.0f} | +{improvement:.1f}%",
                True,
                BLACK
            )
            self.screen.blit(revenue_text, (40, 400))
    
    def draw(self):
        """Draw everything on screen"""
        self.screen.fill(WHITE)
        
        # Draw UI components
        self.draw_ui()
        
        # Draw plots
        self.draw_plots()
        
        pygame.display.flip()
    
    def run(self):
        """Main application loop"""
        while self.running:
            self.process_pending_evaluation()
            self.handle_events()
            self.draw()
            self.clock.tick(60)
        
        pygame.quit()


def main():
    """Main function"""
    try:
        app = PricingApp()
        app.run()
    except Exception as e:
        print(f"Application error: {e}")
        pygame.quit()


if __name__ == "__main__":
    main()
