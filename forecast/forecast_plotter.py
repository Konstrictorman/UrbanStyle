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
        self.data = np.array(historical_data) if historical_data is not None else None
        self.forecast = np.array(forecast_data) if forecast_data is not None else None
        self.dates = dates
        
        if forecast_dates is None and forecast_data is not None:
            # Generate forecast dates
            last_date = dates[-1] if len(dates) > 0 else datetime.now()
            self.forecast_dates = [last_date + timedelta(weeks=i+1) for i in range(len(forecast_data))]
        else:
            self.forecast_dates = forecast_dates
        
    
    def draw(self, screen):
        """Draw the forecast plot"""
        # Draw background
        pygame.draw.rect(screen, self.bg_color, self.rect)
        pygame.draw.rect(screen, (100, 100, 100), self.rect, 2)
        
        # Draw title
        title_text = self.title_font.render("Sales Forecast", True, self.text_color)
        title_rect = title_text.get_rect(center=(self.rect.centerx, self.rect.y + 20))
        screen.blit(title_text, title_rect)
        
        if self.data is None:
            # Draw "No data" message
            no_data_text = self.font.render("No data available", True, self.text_color)
            no_data_rect = no_data_text.get_rect(center=self.rect.center)
            screen.blit(no_data_text, no_data_rect)
            return
        
        # Calculate data bounds
        if self.forecast is not None and len(self.forecast) > 0:
            all_data = np.concatenate([self.data, self.forecast])
        else:
            all_data = self.data
        
        # Ensure we have valid data bounds
        if len(all_data) == 0:
            y_min, y_max = 0, 100
        else:
            data_min = np.min(all_data)
            data_max = np.max(all_data)
            # Add padding only if data range is not zero
            if data_max > data_min:
                y_min = data_min - (data_max - data_min) * 0.1
                y_max = data_max + (data_max - data_min) * 0.1
            else:
                y_min = data_min - abs(data_min) * 0.1
                y_max = data_max + abs(data_max) * 0.1
        
        # Draw grid
        self.draw_grid(screen, y_min, y_max)
        
        # Draw axes
        self.draw_axes(screen, y_min, y_max)
        
        # Draw historical data
        self.draw_historical_data(screen, y_min, y_max)
        
        # Draw forecast
        self.draw_forecast(screen, y_min, y_max)
        
        # Draw legend
        self.draw_legend(screen)
    
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
            # Use 54 weeks total for consistent labeling
            total_weeks = 54
            
            num_labels = min(8, total_weeks)
            for i in range(num_labels):
                idx = int((i / (num_labels - 1)) * (total_weeks - 1)) if num_labels > 1 else 0
                
                # Display actual week numbers (1-54)
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
            return
        
        # Use actual data length for historical data (should be 54 weeks)
        total_weeks = len(self.data)
        
        points = []
        for i, value in enumerate(self.data):
            x = self.plot_x + (i / (total_weeks - 1)) * self.plot_width
            y = self.plot_y + self.plot_height - ((value - y_min) / (y_max - y_min)) * self.plot_height
            points.append((x, y))
        
        # Draw line
        if len(points) > 1:
            pygame.draw.lines(screen, self.data_color, False, points, 3)
        
        # Draw points
        for point in points:
            pygame.draw.circle(screen, self.data_color, point, 4)
    
    def draw_forecast(self, screen, y_min, y_max):
        """Draw forecast line for weeks 47-54"""
        if self.forecast is None or len(self.forecast) < 1:
            return
        
        # Forecast should start at week 47 (index 46) and go to week 54 (index 53)
        # Historical data goes from week 1 to 54, so forecast starts at index 46
        forecast_start_week = 46  # Week 47 in 0-based indexing
        forecast_end_week = 53    # Week 54 in 0-based indexing
        
        points = []
        for i, value in enumerate(self.forecast):
            # Calculate the week index for this forecast point
            week_idx = forecast_start_week + i
            
            # Position on x-axis based on the total 54 weeks
            total_weeks = 54
            x = self.plot_x + (week_idx / (total_weeks - 1)) * self.plot_width
            y = self.plot_y + self.plot_height - ((value - y_min) / (y_max - y_min)) * self.plot_height
            points.append((x, y))
        
        # Draw line
        if len(points) > 1:
            pygame.draw.lines(screen, self.forecast_color, False, points, 3)
        
        # Draw points
        for point in points:
            pygame.draw.circle(screen, self.forecast_color, point, 4)
        
        # Draw connection line between historical and forecast
        if len(self.data) > 0 and len(points) > 0:
            # Connect to the last historical data point (week 46)
            last_hist_idx = 45  # Week 46 in 0-based indexing
            last_hist_x = self.plot_x + (last_hist_idx / (total_weeks - 1)) * self.plot_width
            last_hist_y = self.plot_y + self.plot_height - ((self.data[last_hist_idx] - y_min) / (y_max - y_min)) * self.plot_height
            pygame.draw.line(screen, self.forecast_color, 
                           (last_hist_x, last_hist_y), points[0], 2)
    
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
