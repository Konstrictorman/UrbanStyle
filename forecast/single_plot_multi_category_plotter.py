import pygame
import numpy as np
from datetime import datetime, timedelta

class SinglePlotMultiCategoryPlotter:
    """Single plot area showing all categories with different colored lines"""
    
    def __init__(self, x, y, width, height, categories):
        self.rect = pygame.Rect(x, y, width, height)
        self.categories = categories
        
        # Define colors for each category
        self.category_colors = {
            'Electronics': (255, 0, 0),      # Red
            'Clothing': (0, 255, 0),         # Green  
            'Beauty': (0, 0, 255),           # Blue
            'Baby Stuff': (255, 165, 0),     # Orange
            'Sports': (128, 0, 128)          # Purple
        }
        
        # Define forecast colors (darker versions for contrast)
        self.forecast_colors = {
            'Electronics': (180, 0, 0),      # Dark Red
            'Clothing': (0, 180, 0),         # Dark Green  
            'Beauty': (0, 0, 180),           # Dark Blue
            'Baby Stuff': (200, 100, 0),     # Dark Orange
            'Sports': (100, 0, 100)          # Dark Purple
        }
        
        # Plot dimensions
        self.plot_x = x + 60
        self.plot_y = y + 40
        self.plot_width = width - 80
        self.plot_height = height - 80  # Restored to original since legend is removed
        
        # Font for labels
        self.font = pygame.font.Font(None, 18)
        self.title_font = pygame.font.Font(None, 24)
        self.text_color = (255, 255, 255)  # White text for black background
        
        # Data storage
        self.category_data = {}
        
    def set_data(self, category_data_dict, selected_categories=None):
        """Set data for all categories, optionally filtered by selected categories"""
        if selected_categories is None:
            self.category_data = category_data_dict
        else:
            # Filter data to only include selected categories
            self.category_data = {k: v for k, v in category_data_dict.items() if k in selected_categories}
        
    def draw(self, screen):
        """Draw the single plot with all categories"""
        # Draw background (black)
        pygame.draw.rect(screen, (0, 0, 0), self.rect)
        pygame.draw.rect(screen, (255, 255, 255), self.rect, 2)
        
        # Draw title in top right corner
        title_text = self.title_font.render("Sales Forecast by Category", True, self.text_color)
        title_width = title_text.get_width()
        screen.blit(title_text, (self.rect.x + self.rect.width - title_width - 10, self.rect.y + 10))
        
        # Draw plot area
        self.draw_plot_area(screen)
        
        # Legend removed to give more space for data visualization
        
    def draw_plot_area(self, screen):
        """Draw the main plot area with axes and data"""
        # Draw plot background (black)
        pygame.draw.rect(screen, (0, 0, 0), 
                        (self.plot_x, self.plot_y, self.plot_width, self.plot_height))
        pygame.draw.rect(screen, (255, 255, 255), 
                        (self.plot_x, self.plot_y, self.plot_width, self.plot_height), 2)
        
        # Draw 5x5 grid
        self.draw_grid(screen)
        
        if not self.category_data:
            return
            
        # Find global min/max across all categories
        all_data = []
        for category, data in self.category_data.items():
            if data and 'historical_data' in data and data['historical_data'] is not None:
                all_data.extend(data['historical_data'])
            if data and 'forecasts' in data and data['forecasts'] is not None:
                all_data.extend(data['forecasts'])
        
        if not all_data:
            return
            
        y_min = min(all_data)
        y_max = max(all_data)
        
        # Add padding but ensure y_min is not negative
        y_range = y_max - y_min
        y_min = max(0, y_min - y_range * 0.1)  # Don't go below 0
        y_max += y_range * 0.1
        
        # Draw axes
        self.draw_axes(screen, y_min, y_max)
        
        # Draw data for each category
        for category, data in self.category_data.items():
            if data:
                self.draw_category_data(screen, category, data, y_min, y_max)
    
    def draw_axes(self, screen, y_min, y_max):
        """Draw plot axes and labels"""
        # Y-axis labels
        num_y_labels = 6
        for i in range(num_y_labels):
            value = y_min + (y_max - y_min) * i / (num_y_labels - 1)
            label_text = self.font.render(f"{value:.0f}", True, self.text_color)
            y_pos = self.plot_y + self.plot_height - (i / (num_y_labels - 1)) * self.plot_height
            screen.blit(label_text, (self.plot_x - 50, y_pos - 8))
        
        # X-axis labels (week numbers)
        num_x_labels = 8
        for i in range(num_x_labels):
            week_num = int(1 + (53 * i / (num_x_labels - 1)))
            label_text = self.font.render(f"Week {week_num}", True, self.text_color)
            x_pos = self.plot_x + (i / (num_x_labels - 1)) * self.plot_width
            screen.blit(label_text, (x_pos - 20, self.plot_y + self.plot_height + 10))
        
        # Axis titles
        y_title = self.font.render("Sales Quantity", True, self.text_color)
        screen.blit(y_title, (self.plot_x - 45, self.plot_y - 20))
        
        x_title = self.font.render("Week Number", True, self.text_color)
        screen.blit(x_title, (self.plot_x + self.plot_width // 2 - 40, self.plot_y + self.plot_height + 30))
    
    def draw_category_data(self, screen, category, data, y_min, y_max):
        """Draw historical and forecast data for a specific category"""
        historical_color = self.category_colors.get(category, (0, 0, 0))
        forecast_color = self.forecast_colors.get(category, (0, 0, 0))
        
        # Draw historical data
        if 'historical_data' in data and data['historical_data'] is not None:
            self.draw_line(screen, data['historical_data'], historical_color, y_min, y_max, is_forecast=False)
        
        # Draw forecast data
        if 'forecasts' in data and data['forecasts'] is not None:
            self.draw_line(screen, data['forecasts'], forecast_color, y_min, y_max, is_forecast=True)
    
    def draw_line(self, screen, data, color, y_min, y_max, is_forecast=False):
        """Draw a line for historical or forecast data with markers"""
        if len(data) < 2:
            return
            
        points = []
        total_weeks = 54  # Total weeks in the dataset
        
        for i, value in enumerate(data):
            if is_forecast:
                # Forecast starts at week 47 (index 46)
                week_idx = 46 + i
            else:
                # Historical data goes from week 1 to 54
                week_idx = i
            
            x = self.plot_x + (week_idx / (total_weeks - 1)) * self.plot_width
            y = self.plot_y + self.plot_height - ((value - y_min) / (y_max - y_min)) * self.plot_height
            points.append((x, y))
        
        # Draw the line
        if len(points) > 1:
            pygame.draw.lines(screen, color, False, points, 2)
        
        # Draw markers at each data point
        marker_size = 4
        for point in points:
            x, y = point
            if is_forecast:
                # Use squares for forecast data
                pygame.draw.rect(screen, color, (x - marker_size//2, y - marker_size//2, marker_size, marker_size))
            else:
                # Use circles for historical data
                pygame.draw.circle(screen, color, (int(x), int(y)), marker_size)
    
    def draw_legend(self, screen):
        """Draw legend showing category colors for both historical and forecast data"""
        legend_x = self.rect.x + 10
        legend_y = self.rect.y + self.rect.height - 80
        
        # Draw legend title
        title_text = self.font.render("Legend:", True, self.text_color)
        screen.blit(title_text, (legend_x, legend_y - 20))
        
        # Draw historical data legend (circles) - arrange in 2 rows
        hist_text = self.font.render("Historical (circles):", True, self.text_color)
        screen.blit(hist_text, (legend_x, legend_y))
        
        # First row: 3 categories
        for i, category in enumerate(self.categories[:3]):
            color = self.category_colors.get(category, (0, 0, 0))
            
            # Draw circle marker
            pygame.draw.circle(screen, color, (legend_x + i * 100 + 8, legend_y + 15), 4)
            
            # Draw category name
            name_text = self.font.render(category, True, self.text_color)
            screen.blit(name_text, (legend_x + i * 100 + 15, legend_y + 12))
        
        # Second row: remaining categories
        if len(self.categories) > 3:
            for i, category in enumerate(self.categories[3:]):
                color = self.category_colors.get(category, (0, 0, 0))
                
                # Draw circle marker
                pygame.draw.circle(screen, color, (legend_x + i * 100 + 8, legend_y + 35), 4)
                
                # Draw category name
                name_text = self.font.render(category, True, self.text_color)
                screen.blit(name_text, (legend_x + i * 100 + 15, legend_y + 32))
        
        # Draw forecast data legend (squares) - arrange in 2 rows
        forecast_y = legend_y + 50
        forecast_text = self.font.render("Forecast (squares):", True, self.text_color)
        screen.blit(forecast_text, (legend_x, forecast_y))
        
        # First row: 3 categories
        for i, category in enumerate(self.categories[:3]):
            color = self.forecast_colors.get(category, (0, 0, 0))
            
            # Draw square marker
            pygame.draw.rect(screen, color, (legend_x + i * 100 + 4, forecast_y + 7, 8, 8))
            
            # Draw category name
            name_text = self.font.render(category, True, self.text_color)
            screen.blit(name_text, (legend_x + i * 100 + 15, forecast_y + 4))
        
        # Second row: remaining categories
        if len(self.categories) > 3:
            for i, category in enumerate(self.categories[3:]):
                color = self.forecast_colors.get(category, (0, 0, 0))
                
                # Draw square marker
                pygame.draw.rect(screen, color, (legend_x + i * 100 + 4, forecast_y + 27, 8, 8))
                
                # Draw category name
                name_text = self.font.render(category, True, self.text_color)
                screen.blit(name_text, (legend_x + i * 100 + 15, forecast_y + 24))
    
    def draw_grid(self, screen):
        """Draw a 52x52 grid on the plotting area"""
        grid_color = (30, 30, 30)  # Very dark gray grid lines for fine grid
        
        # Draw vertical grid lines (53 lines for 52 columns)
        for i in range(53):  # 53 lines for 52 columns
            x = self.plot_x + (i / 52) * self.plot_width
            pygame.draw.line(screen, grid_color, 
                           (x, self.plot_y), (x, self.plot_y + self.plot_height), 1)
        
        # Draw horizontal grid lines (53 lines for 52 rows)
        for i in range(53):  # 53 lines for 52 rows
            y = self.plot_y + (i / 52) * self.plot_height
            pygame.draw.line(screen, grid_color, 
                           (self.plot_x, y), (self.plot_x + self.plot_width, y), 1)
