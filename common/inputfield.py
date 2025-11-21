"""
InputField class for pygame GUI elements
"""
import pygame

# Colors (matching store.py)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
LIGHT_GRAY = (200, 200, 200)

class InputField:
    def __init__(self, x, y, width, height, label, initial_value=0.0, min_val=0.0, max_val=1.0, is_integer=False):
        self.rect = pygame.Rect(x, y, width, height)
        self.label = label
        self.value = initial_value
        self.text = str(initial_value)
        self.active = False
        self.font = pygame.font.Font(None, 20)
        self.min_val = min_val
        self.max_val = max_val
        self.is_integer = is_integer
        
    def draw(self, screen):
        # Draw label (closer to the input field)
        label_surface = self.font.render(self.label, True, BLACK)
        screen.blit(label_surface, (self.rect.x, self.rect.y - 20))
        
        # Draw input field background
        color = WHITE if self.active else LIGHT_GRAY
        pygame.draw.rect(screen, color, self.rect)
        
        # Draw border with different colors for active/inactive
        border_color = (0, 0, 255) if self.active else BLACK  # Blue when active, black when inactive
        pygame.draw.rect(screen, border_color, self.rect, 2)
        
        # Draw text
        text_surface = self.font.render(self.text, True, BLACK)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)
        
        # Draw cursor when active
        if self.active:
            cursor_x = self.rect.x + 5 + text_surface.get_width()
            pygame.draw.line(screen, BLACK, (cursor_x, self.rect.y + 3), (cursor_x, self.rect.bottom - 3), 2)
        
    def handle_click(self, pos):
        if pos is not None:
            self.active = self.rect.collidepoint(pos)
        return self.active
        
    def handle_input(self, event):
        if not self.active:
            return False
            
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.key == pygame.K_RETURN:
                self.update_value()
                self.active = False
                return True
            elif event.key == pygame.K_ESCAPE:
                # Cancel editing and reset to original value
                self.text = str(self.value)
                self.active = False
                return False
            elif event.unicode.isdigit() or (event.unicode == '.' and not self.is_integer):
                # Allow digits, and decimal point only for non-integer fields
                if len(self.text) < 6:  # Limit input length
                    self.text += event.unicode
        return False
        
    def update_value(self):
        try:
            if self.is_integer:
                new_value = int(self.text)
            else:
                new_value = float(self.text)
            
            # Validate range using instance variables
            if self.min_val <= new_value <= self.max_val:
                self.value = new_value
            else:
                # Reset to previous value if invalid
                self.text = str(self.value)
        except ValueError:
            # Reset to previous value if invalid
            self.text = str(self.value)

