import pygame

class Checkbox:
    """A simple checkbox widget for Pygame"""
    
    def __init__(self, x, y, size=20, checked=False, label="", font_size=16):
        self.rect = pygame.Rect(x, y, size, size)
        self.checked = checked
        self.label = label
        self.font = pygame.font.Font(None, font_size)
        
        # Colors
        self.bg_color = (255, 255, 255)
        self.border_color = (0, 0, 0)
        self.check_color = (0, 150, 0)
        self.text_color = (0, 0, 0)
        
    def handle_event(self, event):
        """Handle mouse click events"""
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                if self.rect.collidepoint(event.pos):
                    self.checked = not self.checked
                    return True
        return False
    
    def draw(self, screen):
        """Draw the checkbox"""
        # Draw checkbox background
        pygame.draw.rect(screen, self.bg_color, self.rect)
        pygame.draw.rect(screen, self.border_color, self.rect, 2)
        
        # Draw checkmark if checked
        if self.checked:
            # Draw a simple checkmark
            points = [
                (self.rect.x + 4, self.rect.y + self.rect.height // 2),
                (self.rect.x + self.rect.width // 2, self.rect.y + self.rect.height - 4),
                (self.rect.x + self.rect.width - 4, self.rect.y + 4)
            ]
            pygame.draw.lines(screen, self.check_color, False, points, 2)
        
        # Draw label
        if self.label:
            label_text = self.font.render(self.label, True, self.text_color)
            label_x = self.rect.x + self.rect.width + 8
            label_y = self.rect.y + (self.rect.height - label_text.get_height()) // 2
            screen.blit(label_text, (label_x, label_y))
    
    def is_checked(self):
        """Return whether the checkbox is checked"""
        return self.checked
    
    def set_checked(self, checked):
        """Set the checked state"""
        self.checked = checked
