import pygame
import numpy as np
from datetime import datetime, timedelta
from forecast_plotter import ForecastPlotter

class MultiCategoryPlotter:
    """Pygame-based plotter for multiple category sales forecasts"""
    
    def __init__(self, x, y, width, height, categories):
        self.rect = pygame.Rect(x, y, width, height)
        self.categories = categories
        self.plotters = {}
        
        # Calculate plot dimensions for each category
        self.plot_width = width
        self.plot_height = height // len(categories)  # Divide height equally
        
        # Create individual plotters for each category
        for i, category in enumerate(categories):
            plotter_y = y + i * self.plot_height
            self.plotters[category] = ForecastPlotter(x, plotter_y, self.plot_width, self.plot_height)
    
    def set_data(self, category_data_dict):
        """Set data for all categories"""
        for category, data in category_data_dict.items():
            if category in self.plotters:
                self.plotters[category].set_data(
                    data['historical_data'],
                    data['forecasts'],
                    data['dates'],
                    data['forecast_dates']
                )
    
    def draw(self, screen):
        """Draw all category plots"""
        # Draw background
        pygame.draw.rect(screen, (255, 255, 255), self.rect)
        pygame.draw.rect(screen, (0, 0, 0), self.rect, 2)
        
        # Draw category titles and plots
        for i, category in enumerate(self.categories):
            if category in self.plotters:
                # Draw category title
                font = pygame.font.Font(None, 24)
                title_text = font.render(f"{category} Sales", True, (0, 0, 0))
                title_x = self.rect.x + 10
                title_y = self.rect.y + i * self.plot_height + 5
                screen.blit(title_text, (title_x, title_y))
                
                # Draw the plot for this category
                self.plotters[category].draw(screen)
    
    def draw_metrics(self, screen, category_metrics):
        """Draw metrics for all categories"""
        for category, metrics in category_metrics.items():
            if category in self.plotters:
                self.plotters[category].draw_metrics(screen, metrics['mae'], metrics['rmse'])
