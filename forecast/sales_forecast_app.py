import pygame
import sys
import os
import numpy as np
from datetime import datetime, timedelta

# Add the parent directory to the path to import common components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our custom modules
from common.slider import SliderPanel
from common.button import Button
from forecast_plotter import ForecastPlotter
from sales_trainer import SalesTrainer

# Initialize pygame
pygame.init()

# Constants
WINDOW_WIDTH = 1420
WINDOW_HEIGHT = 900
FPS = 60

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
LIGHT_GRAY = (220, 220, 220)
GREEN = (0, 200, 0)
RED = (200, 0, 0)
BLUE = (0, 0, 200)
ORANGE = (255, 165, 0)

class SalesForecastApp:
    """Main application for sales forecasting"""
    
    # Configuration variables - easily changeable at the top of the class
    CSV_FILENAME = "Kagle_.csv"  # Fixed: actual filename with underscore
    
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Sales Forecasting with LSTM - Category-Based")
        self.clock = pygame.time.Clock()
        
        # Initialize data processor to get categories
        csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", self.CSV_FILENAME)
        from sales_data_processor import SalesDataProcessor
        self.processor = SalesDataProcessor(csv_path)
        self.processor.load_and_preprocess()
        self.categories = self.processor.get_categories()
        
        print(f"Available categories: {self.categories}")
        
        # Initialize trainer for selected category (default to first category)
        self.selected_category = self.categories[0] if self.categories else None
        self.trainer = SalesTrainer(csv_path, category=self.selected_category)
        
        # Initialize components first
        self.setup_ui()
        
        # Automatically prepare data on startup
        print(f"Preparing data for category: {self.selected_category}...")
        self.prepare_data()
        
        # Application state
        self.running = True
        self.current_forecast = None
        
    def setup_ui(self):
        """Setup the user interface components"""
        
        # Create slider panel for model parameters (wider to fit slider values)
        self.slider_panel = SliderPanel(20, 20, 420, 520)
        
        # Add sliders for model parameterswqq
        self.slider_panel.add_slider(
            min_val=4, max_val=12, initial_val=8, 
            label="Sequence Length", is_integer=True
        )
        self.slider_panel.add_slider(
            min_val=32, max_val=128, initial_val=64, 
            label="Hidden Size", is_integer=True
        )
        self.slider_panel.add_slider(
            min_val=1, max_val=4, initial_val=2, 
            label="LSTM Layers", is_integer=True
        )
        self.slider_panel.add_slider(
            min_val=0.0, max_val=0.5, initial_val=0.2, 
            label="Dropout Rate"
        )
        self.slider_panel.add_slider(
            min_val=0.0001, max_val=0.01, initial_val=0.001, 
            label="Learning Rate"
        )
        self.slider_panel.add_slider(
            min_val=10, max_val=100, initial_val=50, 
            label="Epochs", is_integer=True
        )
        
        # Create buttons (moved 20px down from previous position)
        button_y = 560
        button_width = 160  # Wider buttons to accommodate text properly
        button_height = 35
        button_spacing = 170  # More spacing between buttons
        
        self.train_button = Button(
            20, button_y, button_width, button_height,
            "Train Model", BLUE, enabled=True,
            on_click=self.on_train_model
        )
        
        self.forecast_button = Button(
            20 + button_spacing, button_y, button_width, button_height,
            "Generate Forecast", ORANGE, enabled=False,
            on_click=self.on_generate_forecast
        )
        
        # Create forecast plotter (adjusted for wider control panel)
        plotter_x = 460
        plotter_y = 20
        plotter_width = WINDOW_WIDTH - 480
        plotter_height = WINDOW_HEIGHT - 40
        print(f"Creating plotter at ({plotter_x}, {plotter_y}) with size ({plotter_width}, {plotter_height})")
        self.plotter = ForecastPlotter(plotter_x, plotter_y, plotter_width, plotter_height)
        print(f"Plotter created with rect: {self.plotter.rect}")
        
        # Category selection buttons
        self.category_buttons = []
        category_button_y = 620
        category_button_width = 80
        category_button_height = 30
        category_button_spacing = 90
        
        for i, category in enumerate(self.categories):
            button_x = 20 + (i % 3) * category_button_spacing
            button_y = category_button_y + (i // 3) * (category_button_height + 5)
            is_selected = (category == self.selected_category)
            button_color = GREEN if is_selected else BLUE
            # Create a closure to capture the category properly
            def make_category_callback(cat):
                return lambda btn: self.on_category_selected(cat)
            button = Button(
                button_x, button_y, category_button_width, category_button_height,
                category[:8], button_color, enabled=True,
                on_click=make_category_callback(category)
            )
            self.category_buttons.append((category, button))
        
        # Status display area (wider to match the control panel)
        self.status_rect = pygame.Rect(20, 700, 420, 125)
    
    # Button event callbacks
    def on_train_model(self, button):
        """Callback for train model button"""
        self.train_model()
    
    def on_generate_forecast(self, button):
        """Callback for generate forecast button"""
        self.generate_forecast()
    
    def on_category_selected(self, category):
        """Callback for category selection"""
        # Validate category is a string
        if not isinstance(category, str):
            print(f"Error: Invalid category type: {type(category)}, value: {category}")
            return
        
        if category == self.selected_category:
            return  # Already selected
        
        print(f"Switching to category: {category}")
        self.selected_category = category
        
        # Update button colors and recalculate
        for cat, button in self.category_buttons:
            if cat == category:
                button.color = GREEN
                button.text_color = button._get_contrasting_color(GREEN)
            else:
                button.color = BLUE
                button.text_color = button._get_contrasting_color(BLUE)
            # Recalculate colors after changing button.color
            button._calculate_colors()
        
        try:
            # Reinitialize trainer with new category
            csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", self.CSV_FILENAME)
            self.trainer = SalesTrainer(csv_path, category=self.selected_category)
            
            # Update parameters from sliders
            values = self.slider_panel.get_values()
            self.trainer.update_parameters(
                sequence_length=int(values[0]),
                hidden_size=int(values[1]),
                num_layers=int(values[2]),
                dropout=values[3],
                learning_rate=values[4],
                epochs=int(values[5])
            )
            
            # Prepare data for new category
            self.prepare_data()
            
            # Reset forecast
            self.current_forecast = None
            self.forecast_button.enabled = False
        except Exception as e:
            print(f"Error switching to category {category}: {e}")
            import traceback
            traceback.print_exc()
    
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            # Handle slider events
            self.slider_panel.handle_event(event)
        
        # Get mouse state for button handling
        mouse_pos = pygame.mouse.get_pos()
        mouse_pressed = pygame.mouse.get_pressed()[0]
        
        # Handle all button events using the new comprehensive method
        self.train_button.handle_events(mouse_pos, mouse_pressed)
        self.forecast_button.handle_events(mouse_pos, mouse_pressed)
        
        # Handle category buttons
        for category, button in self.category_buttons:
            button.handle_events(mouse_pos, mouse_pressed)
    
    def prepare_data(self):
        """Prepare the data for training"""
        # Update parameters from sliders
        values = self.slider_panel.get_values()
        self.trainer.update_parameters(
            sequence_length=int(values[0]),
            hidden_size=int(values[1]),
            num_layers=int(values[2]),
            dropout=values[3],
            learning_rate=values[4],
            epochs=int(values[5])
        )
        
        # Prepare data
        success = self.trainer.prepare_data()
        
        if success:
            success = self.trainer.initialize_model()
            # Don't set plotter data here - wait until forecast is generated
        
        # Enable/disable buttons based on success
        self.train_button.enabled = success
        self.forecast_button.enabled = False
        
        # Clear plotter - don't show data until forecast is generated
        self.plotter.set_data(None, None, None, None)
        self.plotter.title = f"Sales Forecast - {self.selected_category}"
    
    def train_model(self):
        """Start training the model"""
        if not self.train_button.enabled:
            return
        
        # Disable buttons during training
        self.train_button.enabled = False
        self.forecast_button.enabled = False
        
        # Start training
        self.trainer.train_model_async()
    
    def generate_forecast(self):
        """Generate sales forecast (4 weeks ahead for selected category)"""
        if not self.forecast_button.enabled:
            print("Forecast button is disabled. Train model first.")
            return
        
        print(f"Generating forecast for {self.selected_category}...")
        
        # Generate forecast for 4 weeks ahead
        forecast_data = self.trainer.generate_forecast(weeks_ahead=4)
        
        if forecast_data is not None:
            self.current_forecast = forecast_data
            
            print(f"Forecast generated: {len(forecast_data['forecasts'])} weeks")
            print(f"Historical data: {len(forecast_data['historical_data'])} weeks")
            
            # Update plotter with category-specific data
            self.plotter.set_data(
                forecast_data['historical_data'],
                forecast_data['forecasts'],
                forecast_data['dates'],
                forecast_data['forecast_dates']
            )
            
            # Update plotter title to show category
            self.plotter.title = f"Sales Forecast - {self.selected_category}"
            
            print(f"Plotter updated with historical and forecast data")
            
            # Draw metrics if available
            status = self.trainer.get_training_status()
            if status['results'] is not None:
                self.plotter.draw_metrics(
                    self.screen, 
                    status['results']['mae'], 
                    status['results']['rmse']
                )
        else:
            print("Error: Failed to generate forecast data")
    
    def update(self):
        """Update application state"""
        # Check training status
        status = self.trainer.get_training_status()
        
        # Update button states based on training status
        if not status['is_training']:
            if status['status'] == "Training completed":
                self.forecast_button.enabled = True
                self.train_button.enabled = True
                self.train_button.text = "Retrain Model"
                # Don't set plotter data here - wait for "Generate Forecast" button
            elif status['status'] == "Data prepared":
                self.train_button.enabled = True
            elif "error" in status['status'].lower():
                self.train_button.enabled = True
                self.train_button.text = "Train Model"
    
    def draw(self):
        """Draw the application"""
        self.screen.fill(WHITE)
        
        # Draw slider panel
        self.slider_panel.draw(self.screen)
        
        # Draw buttons
        self.train_button.draw(self.screen)
        self.forecast_button.draw(self.screen)
        
        # Draw category buttons
        for category, button in self.category_buttons:
            button.draw(self.screen)
        
        # Draw category label
        font = pygame.font.Font(None, 18)
        category_label = font.render(f"Category: {self.selected_category}", True, BLACK)
        self.screen.blit(category_label, (20, 600))
        
        # Draw status area
        pygame.draw.rect(self.screen, LIGHT_GRAY, self.status_rect)
        pygame.draw.rect(self.screen, BLACK, self.status_rect, 2)
        
        # Draw status text
        status = self.trainer.get_training_status()
        
        status_texts = [
            f"Category: {self.selected_category}",
            f"Status: {status['status']}",
            f"Progress: {status['progress']:.1%}",
            f"Training: {'Yes' if status['is_training'] else 'No'}"
        ]
        
        if status['results'] is not None:
            status_texts.extend([
                f"MAE: {status['results']['mae']:.2f}",
                f"RMSE: {status['results']['rmse']:.2f}"
            ])
        
        y_offset = 10
        for text in status_texts:
            text_surface = font.render(text, True, BLACK)
            self.screen.blit(text_surface, (self.status_rect.x + 10, self.status_rect.y + y_offset))
            y_offset += 18
        
        # Draw forecast plot
        try:
            if hasattr(self, 'plotter') and self.plotter is not None:
                self.plotter.draw(self.screen)
            else:
                print("ERROR: Plotter is None or doesn't exist!")
                # Draw a placeholder rectangle
                placeholder_rect = pygame.Rect(460, 20, WINDOW_WIDTH - 480, WINDOW_HEIGHT - 40)
                pygame.draw.rect(self.screen, (100, 100, 100), placeholder_rect)
                error_text = pygame.font.Font(None, 24).render("Plotter Error", True, (255, 0, 0))
                self.screen.blit(error_text, (470, 30))
        except Exception as e:
            print(f"ERROR drawing plotter: {e}")
            import traceback
            traceback.print_exc()
            # Draw error rectangle
            error_rect = pygame.Rect(460, 20, WINDOW_WIDTH - 480, WINDOW_HEIGHT - 40)
            pygame.draw.rect(self.screen, (255, 0, 0), error_rect)
            error_text = pygame.font.Font(None, 24).render(f"Plot Error: {str(e)}", True, (255, 255, 255))
            self.screen.blit(error_text, (470, 30))
        
        # Draw training progress bar if training
        if status['is_training']:
            self.draw_progress_bar(status['progress'])
        
        pygame.display.flip()
    
    def draw_progress_bar(self, progress):
        """Draw training progress bar"""
        bar_rect = pygame.Rect(20, 640, 300, 20)
        
        # Background
        pygame.draw.rect(self.screen, LIGHT_GRAY, bar_rect)
        pygame.draw.rect(self.screen, BLACK, bar_rect, 2)
        
        # Progress fill
        fill_width = int(bar_rect.width * progress)
        fill_rect = pygame.Rect(bar_rect.x, bar_rect.y, fill_width, bar_rect.height)
        pygame.draw.rect(self.screen, GREEN, fill_rect)
        
        # Progress text
        font = pygame.font.Font(None, 16)
        progress_text = font.render(f"{progress:.1%}", True, BLACK)
        text_rect = progress_text.get_rect(center=bar_rect.center)
        self.screen.blit(progress_text, text_rect)
    
    def run(self):
        """Main application loop"""
        print("Sales Forecasting Application Started")
        print("Instructions:")
        print("1. Adjust model parameters using sliders")
        print("2. Click 'Prepare Data' to load and preprocess data")
        print("3. Click 'Train Model' to train the LSTM")
        print("4. Click 'Generate Forecast' to see predictions")
        
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)
        
        pygame.quit()
        sys.exit()

def main():
    """Main function"""
    try:
        app = SalesForecastApp()
        app.run()
    except Exception as e:
        print(f"Application error: {e}")
        pygame.quit()
        sys.exit(1)

if __name__ == "__main__":
    main()
