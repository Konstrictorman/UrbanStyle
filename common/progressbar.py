"""
ProgressBar class for pygame GUI elements
"""
import pygame

# Colors (matching store.py)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
LIGHT_GRAY = (200, 200, 200)

class ProgressBar:
    def __init__(self, x, y, width, height, max_value=100):
        self.rect = pygame.Rect(x, y, width, height)
        self.max_value = max_value
        self.current_value = 0
        
    def update(self, value):
        self.current_value = min(value, self.max_value)
        
    def draw(self, screen):
        # Background
        pygame.draw.rect(screen, LIGHT_GRAY, self.rect)
        pygame.draw.rect(screen, BLACK, self.rect, 2)
        
        # Progress fill - fill according to percentage
        if self.max_value > 0:
            fill_width = int((self.current_value / self.max_value) * self.rect.width)
            if fill_width > 0:
                fill_rect = pygame.Rect(self.rect.x, self.rect.y, fill_width, self.rect.height)
                pygame.draw.rect(screen, GREEN, fill_rect)

