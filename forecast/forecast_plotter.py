import pygame
import numpy as np
from datetime import datetime, timedelta

class ForecastPlotter:
    """Pygame-based plotter for sales forecasts"""
    
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)
        self.data = None
        self.forecast = None
        self.test_data = None  # Actual values for test period
        self.dates = None
        self.forecast_dates = None
        self.title = "Sales Forecast"  # Default title, can be updated
        
        # Colors
        self.bg_color = (0, 0, 0)  # Black background
        self.grid_color = (50, 50, 50)  # Dark gray grid
        self.data_color = (0, 100, 200)  # Blue for historical
        self.forecast_color = (200, 50, 50)  # Red for forecast
        self.test_color = (50, 150, 50)  # Green for actual test values
        self.text_color = (255, 255, 255)  # White text for black background
        self.axis_color = (200, 200, 200)  # Light gray axes
        
        # Fonts
        self.font = pygame.font.Font(None, 16)
        self.title_font = pygame.font.Font(None, 24)
        
        # Margins
        self.margin_left = 60
        self.margin_right = 20
        self.margin_top = 40
        self.margin_bottom = 60
        
        # Plot area
        self.plot_x = x + self.margin_left
        self.plot_y = y + self.margin_top
        self.plot_width = width - self.margin_left - self.margin_right
        self.plot_height = height - self.margin_top - self.margin_bottom
        
    def set_data(self, historical_data, forecast_data, dates, forecast_dates=None):
        """Set the data to plot"""
        print(f"\n=== PLOTTER.SET_DATA CALLED ===")
        print(f"  historical_data: {type(historical_data)}, length: {len(historical_data) if historical_data is not None else 'None'}")
        print(f"  forecast_data: {type(forecast_data)}, length: {len(forecast_data) if forecast_data is not None else 'None'}")
        print(f"  dates: {type(dates)}, length: {len(dates) if dates is not None else 'None'}")
        
        try:
            if historical_data is not None:
                # Convert to numpy array, handling different input types
                if isinstance(historical_data, (list, tuple)):
                    self.data = np.array(historical_data, dtype=np.float64)
                elif isinstance(historical_data, np.ndarray):
                    self.data = historical_data.astype(np.float64)
                else:
                    self.data = np.array([historical_data], dtype=np.float64)
                print(f"  Converted historical_data to array with shape: {self.data.shape}, dtype: {self.data.dtype}")
                print(f"  First few values: {self.data[:5] if len(self.data) >= 5 else self.data}")
            else:
                self.data = None
                print(f"  historical_data is None, setting self.data to None")
            
            if forecast_data is not None:
                # Convert to numpy array, handling different input types
                if isinstance(forecast_data, (list, tuple)):
                    self.forecast = np.array(forecast_data, dtype=np.float64)
                elif isinstance(forecast_data, np.ndarray):
                    self.forecast = forecast_data.astype(np.float64)
                else:
                    self.forecast = np.array([forecast_data], dtype=np.float64)
                print(f"  Converted forecast_data to array with shape: {self.forecast.shape}, dtype: {self.forecast.dtype}")
                print(f"  First few values: {self.forecast[:5] if len(self.forecast) >= 5 else self.forecast}")
            else:
                self.forecast = None
                print(f"  forecast_data is None, setting self.forecast to None")
            
            self.dates = dates
            print(f"=== END SET_DATA ===\n")
        except Exception as e:
            print(f"ERROR in set_data: {e}")
            import traceback
            traceback.print_exc()
            self.data = None
            self.forecast = None
        
        if forecast_dates is None and forecast_data is not None:
            # Generate forecast dates
            last_date = dates[-1] if len(dates) > 0 else datetime.now()
            self.forecast_dates = [last_date + timedelta(weeks=i+1) for i in range(len(forecast_data))]
        else:
            self.forecast_dates = forecast_dates
        
    
    def draw(self, screen):
        """Draw the forecast plot"""
        try:
            # Draw background (always draw this)
            pygame.draw.rect(screen, self.bg_color, self.rect)
            pygame.draw.rect(screen, (100, 100, 100), self.rect, 2)
            
            # Draw title (always draw this)
            title_text = self.title_font.render(self.title, True, self.text_color)
            title_rect = title_text.get_rect(center=(self.rect.centerx, self.rect.y + 20))
            screen.blit(title_text, title_rect)
            
            if self.data is None:
                # Draw "No data" message
                no_data_text = self.font.render("No data available - Train model and generate forecast", True, self.text_color)
                no_data_rect = no_data_text.get_rect(center=self.rect.center)
                screen.blit(no_data_text, no_data_rect)
                return
        except Exception as e:
            # If there's an error, at least draw the background
            print(f"Error in plotter draw: {e}")
            pygame.draw.rect(screen, self.bg_color, self.rect)
            error_text = self.font.render("Error drawing plot", True, self.text_color)
            error_rect = error_text.get_rect(center=self.rect.center)
            screen.blit(error_text, error_rect)
            return
        
        # Calculate data bounds
        print(f"\n=== DRAWING PLOT ===")
        print(f"self.data: {self.data is not None}, length: {len(self.data) if self.data is not None else 0}")
        print(f"self.forecast: {self.forecast is not None}, length: {len(self.forecast) if self.forecast is not None else 0}")
        
        if self.forecast is not None and len(self.forecast) > 0:
            all_data = np.concatenate([self.data, self.forecast])
        else:
            all_data = self.data
        
        # Ensure we have valid data bounds
        if len(all_data) == 0:
            y_min, y_max = 0, 100
            print(f"  No data, using default bounds: y_min={y_min}, y_max={y_max}")
        else:
            data_min = np.min(all_data)
            data_max = np.max(all_data)
            print(f"  Data range: {data_min} to {data_max}")
            # Add padding only if data range is not zero
            if data_max > data_min:
                y_min = data_min - (data_max - data_min) * 0.1
                y_max = data_max + (data_max - data_min) * 0.1
            else:
                y_min = data_min - abs(data_min) * 0.1
                y_max = data_max + abs(data_max) * 0.1
            print(f"  Calculated bounds: y_min={y_min}, y_max={y_max}")
        
        # Draw grid
        self.draw_grid(screen, y_min, y_max)
        
        # Draw axes
        self.draw_axes(screen, y_min, y_max)
        
        # Draw historical data
        try:
            self.draw_historical_data(screen, y_min, y_max)
        except Exception as e:
            print(f"Error drawing historical data: {e}")
            import traceback
            traceback.print_exc()
        
        # Draw forecast
        try:
            self.draw_forecast(screen, y_min, y_max)
        except Exception as e:
            print(f"Error drawing forecast: {e}")
            import traceback
            traceback.print_exc()
        
        # Draw legend
        try:
            self.draw_legend(screen)
        except Exception as e:
            print(f"Error drawing legend: {e}")
            import traceback
            traceback.print_exc()
    
    def draw_grid(self, screen, y_min, y_max):
        """Draw grid lines"""
        # Vertical grid lines
        num_vertical = 10
        for i in range(num_vertical + 1):
            x = self.plot_x + (i / num_vertical) * self.plot_width
            pygame.draw.line(screen, self.grid_color, 
                           (x, self.plot_y), (x, self.plot_y + self.plot_height))
        
        # Horizontal grid lines
        num_horizontal = 8
        for i in range(num_horizontal + 1):
            y = self.plot_y + (i / num_horizontal) * self.plot_height
            pygame.draw.line(screen, self.grid_color,
                           (self.plot_x, y), (self.plot_x + self.plot_width, y))
    
    def draw_axes(self, screen, y_min, y_max):
        """Draw axes and labels"""
        # Draw axes
        pygame.draw.line(screen, self.axis_color,
                        (self.plot_x, self.plot_y + self.plot_height),
                        (self.plot_x + self.plot_width, self.plot_y + self.plot_height), 2)  # X-axis
        pygame.draw.line(screen, self.axis_color,
                        (self.plot_x, self.plot_y),
                        (self.plot_x, self.plot_y + self.plot_height), 2)  # Y-axis
        
        # Y-axis labels
        num_labels = 6
        for i in range(num_labels + 1):
            value = y_min + (i / num_labels) * (y_max - y_min)
            label_text = self.font.render(f"{value:.0f}", True, self.text_color)
            y_pos = self.plot_y + self.plot_height - (i / num_labels) * self.plot_height
            screen.blit(label_text, (self.plot_x - 50, y_pos - 8))
        
        # X-axis labels (week numbers)
        if self.data is not None and len(self.data) > 0:
            # Calculate total weeks including forecast
            historical_weeks = len(self.data)
            forecast_weeks = len(self.forecast) if self.forecast is not None else 0
            total_weeks = historical_weeks + forecast_weeks
            
            num_labels = min(8, total_weeks)
            for i in range(num_labels):
                idx = int((i / (num_labels - 1)) * (total_weeks - 1)) if num_labels > 1 else 0
                
                # Display actual week numbers (1 to total_weeks)
                week_number = idx + 1
                label_text = self.font.render(f"Week {week_number}", True, self.text_color)
                
                x_pos = self.plot_x + (idx / (total_weeks - 1)) * self.plot_width
                screen.blit(label_text, (x_pos - 20, self.plot_y + self.plot_height + 10))
        
        # Axis titles
        y_title = self.font.render("Sales Quantity", True, self.text_color)
        screen.blit(y_title, (self.plot_x - 55, self.plot_y - 30))
        
        x_title = self.font.render("Week Number", True, self.text_color)
        screen.blit(x_title, (self.plot_x + self.plot_width//2 - 40, self.plot_y + self.plot_height + 35))
    
    def draw_historical_data(self, screen, y_min, y_max):
        """Draw historical data line"""
        if self.data is None or len(self.data) < 1:
            print("draw_historical_data: No data to draw")
            return
        
        print(f"draw_historical_data: Drawing {len(self.data)} data points")
        print(f"  y_min: {y_min}, y_max: {y_max}")
        print(f"  data range: {np.min(self.data)} to {np.max(self.data)}")
        
        # Calculate total weeks including forecast for proper scaling
        historical_weeks = len(self.data)
        forecast_weeks = len(self.forecast) if self.forecast is not None else 0
        total_weeks = historical_weeks + forecast_weeks
        
        # Prevent division by zero
        if total_weeks <= 1:
            total_weeks = 2
        if y_max == y_min:
            y_max = y_min + 1  # Prevent division by zero
        
        points = []
        for i, value in enumerate(self.data):
            # Position based on total weeks (historical + forecast)
            x = self.plot_x + (i / (total_weeks - 1)) * self.plot_width
            y = self.plot_y + self.plot_height - ((value - y_min) / (y_max - y_min)) * self.plot_height
            points.append((int(x), int(y)))
        
        print(f"  Generated {len(points)} points, first: {points[0] if points else 'None'}, last: {points[-1] if points else 'None'}")
        
        # Draw line
        if len(points) > 1:
            pygame.draw.lines(screen, self.data_color, False, points, 3)
            print(f"  Drew line with {len(points)} points")
        
        # Draw points
        for point in points:
            pygame.draw.circle(screen, self.data_color, point, 4)
        
        print(f"  Finished drawing historical data")
    
    def draw_forecast(self, screen, y_min, y_max):
        """Draw forecast line for next 4 weeks (after historical data)"""
        if self.forecast is None or len(self.forecast) < 1:
            print("draw_forecast: No forecast data to draw")
            return
        
        if self.data is None or len(self.data) < 1:
            print("draw_forecast: No historical data available")
            return
        
        print(f"draw_forecast: Drawing {len(self.forecast)} forecast points")
        print(f"  y_min: {y_min}, y_max: {y_max}")
        print(f"  forecast range: {np.min(self.forecast)} to {np.max(self.forecast)}")
        
        # Forecast starts AFTER the last historical data point
        # Calculate total weeks including forecast
        historical_weeks = len(self.data)
        forecast_weeks = len(self.forecast)
        total_weeks = historical_weeks + forecast_weeks
        
        # Prevent division by zero
        if total_weeks <= 1:
            total_weeks = 2
        if y_max == y_min:
            y_max = y_min + 1  # Prevent division by zero
        
        points = []
        for i, value in enumerate(self.forecast):
            # Calculate the week index for this forecast point (after historical data)
            week_idx = historical_weeks + i
            
            # Position on x-axis based on total weeks (historical + forecast)
            x = self.plot_x + (week_idx / (total_weeks - 1)) * self.plot_width
            y = self.plot_y + self.plot_height - ((value - y_min) / (y_max - y_min)) * self.plot_height
            points.append((int(x), int(y)))
        
        print(f"  Generated {len(points)} forecast points, first: {points[0] if points else 'None'}, last: {points[-1] if points else 'None'}")
        
        # Draw line
        if len(points) > 1:
            pygame.draw.lines(screen, self.forecast_color, False, points, 3)
            print(f"  Drew forecast line with {len(points)} points")
        
        # Draw points
        for point in points:
            pygame.draw.circle(screen, self.forecast_color, point, 4)
        
        # Draw connection line between historical and forecast
        if len(self.data) > 0 and len(points) > 0:
            # Connect to the last historical data point
            last_hist_idx = len(self.data) - 1
            last_hist_x = int(self.plot_x + (last_hist_idx / (total_weeks - 1)) * self.plot_width)
            last_hist_y = int(self.plot_y + self.plot_height - ((self.data[last_hist_idx] - y_min) / (y_max - y_min)) * self.plot_height)
            pygame.draw.line(screen, self.forecast_color, 
                           (last_hist_x, last_hist_y), points[0], 2)
            print(f"  Drew connection line from ({last_hist_x}, {last_hist_y}) to {points[0]}")
        
        print(f"  Finished drawing forecast")
    
    def draw_test_data(self, screen, y_min, y_max):
        """Draw actual test data (green line) for comparison with forecast"""
        if self.test_data is None or len(self.test_data) < 1:
            return
        
        # Calculate total points for proper scaling (same as forecast method)
        historical_points = len(self.data) if self.data is not None else 0
        forecast_points = len(self.forecast) if self.forecast is not None else 0
        total_points = historical_points + forecast_points
        
        # Calculate starting position (immediately after historical data)
        start_idx = historical_points
        
        points = []
        for i, value in enumerate(self.test_data):
            idx = start_idx + i
            x = self.plot_x + (idx / (total_points - 1)) * self.plot_width
            y = self.plot_y + self.plot_height - ((value - y_min) / (y_max - y_min)) * self.plot_height
            points.append((x, y))
        
        # Draw line
        if len(points) > 1:
            pygame.draw.lines(screen, self.test_color, False, points, 3)
        
        # Draw points
        for point in points:
            pygame.draw.circle(screen, self.test_color, point, 4)
    
    def draw_legend(self, screen):
        """Draw legend"""
        legend_x = self.plot_x + self.plot_width - 150
        legend_y = self.plot_y + 10
        
        # Historical data legend
        pygame.draw.line(screen, self.data_color, 
                        (legend_x, legend_y), (legend_x + 20, legend_y), 3)
        hist_text = self.font.render("Historical", True, self.text_color)
        screen.blit(hist_text, (legend_x + 25, legend_y - 6))
        
        # Forecast legend
        pygame.draw.line(screen, self.forecast_color,
                        (legend_x, legend_y + 20), (legend_x + 20, legend_y + 20), 3)
        forecast_text = self.font.render("Forecast", True, self.text_color)
        screen.blit(forecast_text, (legend_x + 25, legend_y + 14))
    
    def get_metrics_text(self, mae, rmse):
        """Get formatted metrics text"""
        return f"MAE: {mae:.2f} | RMSE: {rmse:.2f}"
    
    def draw_metrics(self, screen, mae, rmse):
        """Draw evaluation metrics"""
        metrics_text = self.get_metrics_text(mae, rmse)
        metrics_surface = self.font.render(metrics_text, True, self.text_color)
        screen.blit(metrics_surface, (self.plot_x, self.plot_y - 25))
