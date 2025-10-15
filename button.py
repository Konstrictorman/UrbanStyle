"""
Enhanced Button class for pygame GUI elements with hover effects, shadows, and beveled styling
"""
import pygame

# Colors (matching store.py)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
LIGHT_GRAY = (220, 220, 220)
DARK_GRAY = (80, 80, 80)

class Button:
    def __init__(self, x, y, width, height, text, color, text_color=None, 
                 shadow_offset=3, bevel_depth=2, hover_scale=1.05, enabled=True):
        self.rect = pygame.Rect(x, y, width, height)
        self.original_rect = pygame.Rect(x, y, width, height)  # Keep original position
        self.text = text
        self.color = color
        
        # Auto-determine text color if not provided
        if text_color is None:
            self.text_color = self._get_contrasting_color(color)
        else:
            self.text_color = text_color
            
        self.font = pygame.font.Font(None, 24)
        
        # Visual effects
        self.shadow_offset = shadow_offset
        self.bevel_depth = bevel_depth
        self.hover_scale = hover_scale
        
        # State tracking
        self.is_hovered = False
        self.is_pressed = False
        self.enabled = enabled
        
        # Calculate colors based on state
        self._calculate_colors()
        
    def _brighten_color(self, color, amount):
        """Brighten a color by the specified amount"""
        return tuple(min(255, c + amount) for c in color)
    
    def _darken_color(self, color, amount):
        """Darken a color by the specified amount"""
        return tuple(max(0, c - amount) for c in color)
    
    def _get_contrasting_color(self, bg_color):
        """Get a contrasting text color based on background brightness"""
        # Calculate luminance
        luminance = (0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2])
        
        # Return black for bright backgrounds, white for dark backgrounds
        return BLACK if luminance > 128 else WHITE
    
    def _calculate_colors(self):
        """Calculate all color variants based on current state"""
        if not self.enabled:
            # Disabled colors (grayed out)
            self.current_color = self._darken_color(self.color, 60)
            self.hover_color = self.current_color
            self.pressed_color = self.current_color
            self.text_color = self._darken_color(self.text_color, 40)
        else:
            # Normal colors
            self.current_color = self.color
            self.hover_color = self._brighten_color(self.color, 20)
            self.pressed_color = self._darken_color(self.color, 30)
    
    def set_enabled(self, enabled):
        """Enable or disable the button"""
        self.enabled = enabled
        self._calculate_colors()
        if not enabled:
            self.is_hovered = False
            self.is_pressed = False
    
    def update_hover(self, mouse_pos):
        """Update hover state based on mouse position"""
        if not self.enabled:
            self.is_hovered = False
            self.rect = self.original_rect.copy()
            return
            
        self.is_hovered = self.rect.collidepoint(mouse_pos)
        
        # Adjust rect for hover effect (slight scaling)
        if self.is_hovered and not self.is_pressed:
            # Scale up slightly
            scale_factor = self.hover_scale
            new_width = int(self.original_rect.width * scale_factor)
            new_height = int(self.original_rect.height * scale_factor)
            
            # Center the scaled rect
            offset_x = (new_width - self.original_rect.width) // 2
            offset_y = (new_height - self.original_rect.height) // 2
            
            self.rect = pygame.Rect(
                self.original_rect.x - offset_x,
                self.original_rect.y - offset_y,
                new_width,
                new_height
            )
        else:
            # Reset to original size
            self.rect = self.original_rect.copy()
    
    def handle_click(self, mouse_pos, mouse_pressed):
        """Handle click events and update pressed state"""
        if not self.enabled:
            return False
            
        if mouse_pressed and self.rect.collidepoint(mouse_pos):
            self.is_pressed = True
            return True
        elif not mouse_pressed:
            self.is_pressed = False
        return False
    
    def draw(self, screen):
        # Determine current color based on state
        if not self.enabled:
            current_color = self.current_color
        elif self.is_pressed:
            current_color = self.pressed_color
        elif self.is_hovered:
            current_color = self.hover_color
        else:
            current_color = self.current_color
        
        # Draw shadow (only if enabled)
        if self.enabled:
            shadow_rect = pygame.Rect(
                self.original_rect.x + self.shadow_offset,
                self.original_rect.y + self.shadow_offset,
                self.original_rect.width,
                self.original_rect.height
            )
            pygame.draw.rect(screen, (0, 0, 0), shadow_rect)
        
        # Draw main button with beveled effect
        pygame.draw.rect(screen, current_color, self.rect)
        
        # Draw beveled border (3D effect) - only if enabled
        if self.enabled and self.bevel_depth > 0:
            # Top and left edges (highlight)
            highlight_color = self._brighten_color(current_color, 40)
            pygame.draw.line(screen, highlight_color, 
                           (self.rect.left, self.rect.top), 
                           (self.rect.right - 1, self.rect.top), self.bevel_depth)
            pygame.draw.line(screen, highlight_color, 
                           (self.rect.left, self.rect.top), 
                           (self.rect.left, self.rect.bottom - 1), self.bevel_depth)
            
            # Bottom and right edges (shadow)
            shadow_edge_color = self._darken_color(current_color, 40)
            pygame.draw.line(screen, shadow_edge_color, 
                           (self.rect.left, self.rect.bottom - 1), 
                           (self.rect.right - 1, self.rect.bottom - 1), self.bevel_depth)
            pygame.draw.line(screen, shadow_edge_color, 
                           (self.rect.right - 1, self.rect.top), 
                           (self.rect.right - 1, self.rect.bottom - 1), self.bevel_depth)
        elif not self.enabled:
            # Draw flat border for disabled state
            pygame.draw.rect(screen, self._darken_color(current_color, 20), self.rect, 2)
        
        # Draw text
        text_surface = self.font.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)
        
        # Draw additional hover effect (glow) - only if enabled
        if self.enabled and self.is_hovered and not self.is_pressed:
            # Draw a subtle glow effect
            glow_rect = pygame.Rect(
                self.rect.x - 2,
                self.rect.y - 2,
                self.rect.width + 4,
                self.rect.height + 4
            )
            pygame.draw.rect(screen, current_color, glow_rect, 2)
    
    def is_clicked(self, pos):
        """Check if button is clicked (uses original rect for consistent click detection)"""
        return self.enabled and self.original_rect.collidepoint(pos)
