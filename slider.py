import pygame
import math

class Slider:
    def __init__(self, x, y, width, height, min_val, max_val, initial_val, label, is_integer=False):
        self.rect = pygame.Rect(x, y, width, height)
        self.min_val = min_val
        self.max_val = max_val
        self.is_integer = is_integer
        
        # Calculate initial position
        if initial_val is None:
            initial_val = (min_val + max_val) / 2
        
        self.value = initial_val
        self.label = label
        
        # Visual properties
        self.bg_color = (200, 200, 200)
        self.slider_color = (100, 150, 200)
        self.handle_color = (50, 100, 150)
        self.text_color = (0, 0, 0)
        
        # Handle properties
        self.handle_width = 20
        self.handle_height = height - 4
        self.handle_rect = pygame.Rect(0, 0, self.handle_width, self.handle_height)
        
        self.font = pygame.font.Font(None, 18)
        self.label_font = pygame.font.Font(None, 20)
        
        self.dragging = False
        self.update_handle_position()
    
    def update_handle_position(self):
        """Update handle position based on current value"""
        # Calculate handle position
        ratio = (self.value - self.min_val) / (self.max_val - self.min_val)
        handle_x = self.rect.x + 2 + ratio * (self.rect.width - 4 - self.handle_width)
        
        self.handle_rect.x = handle_x
        self.handle_rect.y = self.rect.y + 2
    
    def get_value(self):
        """Get current slider value"""
        if self.is_integer:
            return int(round(self.value))
        return self.value
    
    def set_value(self, value):
        """Set slider value"""
        self.value = max(self.min_val, min(self.max_val, value))
        self.update_handle_position()
    
    def handle_event(self, event):
        """Handle pygame events"""
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                mouse_pos = event.pos
                if self.rect.collidepoint(mouse_pos) or self.handle_rect.collidepoint(mouse_pos):
                    self.dragging = True
                    self.update_value_from_mouse(mouse_pos[0])
                    return True
        
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self.dragging = False
        
        elif event.type == pygame.MOUSEMOTION:
            if self.dragging:
                self.update_value_from_mouse(event.pos[0])
                return True
        
        return False
    
    def update_value_from_mouse(self, mouse_x):
        """Update value based on mouse position"""
        # Calculate ratio
        slider_width = self.rect.width - 4 - self.handle_width
        mouse_offset = mouse_x - (self.rect.x + 2)
        ratio = max(0, min(1, mouse_offset / slider_width))
        
        # Update value
        new_value = self.min_val + ratio * (self.max_val - self.min_val)
        self.set_value(new_value)
    
    def draw(self, screen):
        """Draw the slider"""
        # Draw background
        pygame.draw.rect(screen, self.bg_color, self.rect)
        pygame.draw.rect(screen, (150, 150, 150), self.rect, 2)
        
        # Draw slider track
        track_rect = pygame.Rect(self.rect.x + 2, self.rect.centery - 2, 
                               self.rect.width - 4, 4)
        pygame.draw.rect(screen, self.slider_color, track_rect)
        
        # Draw handle
        pygame.draw.rect(screen, self.handle_color, self.handle_rect)
        pygame.draw.rect(screen, (0, 0, 0), self.handle_rect, 2)
        
        # Draw label (positioned much closer to the slider - reduced spacing by half)
        label_text = self.label_font.render(self.label, True, self.text_color)
        screen.blit(label_text, (self.rect.x, self.rect.y - 12))
        
        # Draw value (positioned to the right of the slider with more space)
        value_text = f"{self.get_value():.2f}" if not self.is_integer else str(self.get_value())
        value_surface = self.font.render(value_text, True, self.text_color)
        screen.blit(value_surface, (self.rect.x + self.rect.width + 15, self.rect.y + 5))
        
        # Draw min/max labels
        min_text = self.font.render(f"{self.min_val}", True, self.text_color)
        max_text = self.font.render(f"{self.max_val}", True, self.text_color)
        screen.blit(min_text, (self.rect.x, self.rect.bottom + 5))
        screen.blit(max_text, (self.rect.right - max_text.get_width(), self.rect.bottom + 5))

class SliderPanel:
    """Container for multiple sliders"""
    
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)
        self.sliders = []
        self.bg_color = (240, 240, 240)
        self.border_color = (100, 100, 100)
        
        # Default slider parameters
        self.slider_height = 30
        self.slider_spacing = 70  # Reduced spacing since labels are closer
        
    def add_slider(self, min_val, max_val, initial_val, label, is_integer=False):
        """Add a slider to the panel"""
        slider_y = self.rect.y + 60 + len(self.sliders) * self.slider_spacing  # Moved down 20px (40->60)
        slider_width = self.rect.width - 40
        
        slider = Slider(self.rect.x + 20, slider_y, slider_width, self.slider_height,
                       min_val, max_val, initial_val, label, is_integer)
        self.sliders.append(slider)
        return slider
    
    def handle_event(self, event):
        """Handle events for all sliders"""
        for slider in self.sliders:
            if slider.handle_event(event):
                return True
        return False
    
    def draw(self, screen):
        """Draw the panel and all sliders"""
        # Draw panel background
        pygame.draw.rect(screen, self.bg_color, self.rect)
        pygame.draw.rect(screen, self.border_color, self.rect, 2)
        
        # Draw title
        title_font = pygame.font.Font(None, 24)
        title_text = title_font.render("Model Parameters", True, (0, 0, 0))
        screen.blit(title_text, (self.rect.x + 10, self.rect.y + 10))
        
        # Draw all sliders
        for slider in self.sliders:
            slider.draw(screen)
    
    def get_values(self):
        """Get values from all sliders"""
        return [slider.get_value() for slider in self.sliders]
    
    def get_slider(self, index):
        """Get slider by index"""
        if 0 <= index < len(self.sliders):
            return self.sliders[index]
        return None
