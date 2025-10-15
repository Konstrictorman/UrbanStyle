"""
InputField class for pygame GUI elements
"""
import pygame

# Colors (matching store.py)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
LIGHT_GRAY = (200, 200, 200)

class InputField:
    def __init__(self, x, y, width, height, label, initial_value=0.0):
        self.rect = pygame.Rect(x, y, width, height)
        self.label = label
        self.value = initial_value
        self.text = str(initial_value)
        self.active = False
        self.font = pygame.font.Font(None, 20)
        
    def draw(self, screen):
        # Draw label (closer to the input field)
        label_surface = self.font.render(self.label, True, BLACK)
        screen.blit(label_surface, (self.rect.x, self.rect.y - 20))
        
        # Draw input field background
        color = WHITE if self.active else LIGHT_GRAY
        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, BLACK, self.rect, 2)
        
        # Draw text
        text_surface = self.font.render(self.text, True, BLACK)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)
        
    def handle_click(self, pos):
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
            elif event.unicode.isdigit() or event.unicode == '.':
                # Allow digits and decimal point
                if len(self.text) < 6:  # Limit input length
                    self.text += event.unicode
        return False
        
    def update_value(self):
        try:
            new_value = float(self.text)
            # Validate range
            if 0.0 <= new_value <= 1.0:
                self.value = new_value
            else:
                # Reset to previous value if invalid
                self.text = str(self.value)
        except ValueError:
            # Reset to previous value if invalid
            self.text = str(self.value)

