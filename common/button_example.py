"""
Example demonstrating the enhanced Button class with hover and click events
"""

import pygame
import sys
import os

# Add parent directory to path to import common components
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from common.button import Button

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
FPS = 60

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

class ButtonExample:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Enhanced Button Example")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        
        # Create buttons with event callbacks
        self.start_button = Button(
            100, 100, 150, 50, "Start", GREEN,
            on_hover=self.on_button_hover,
            on_click=self.on_button_click
        )
        
        self.stop_button = Button(
            300, 100, 150, 50, "Stop", RED,
            on_hover=self.on_button_hover,
            on_click=self.on_button_click
        )
        
        self.action_button = Button(
            500, 100, 150, 50, "Action", BLUE,
            on_hover=self.on_button_hover,
            on_click=self.on_button_click
        )
        
        # Status tracking
        self.hover_status = "No hover"
        self.click_status = "No clicks"
        self.click_count = 0
    
    def on_button_hover(self, button, is_hovering):
        """Callback for hover events"""
        if is_hovering:
            self.hover_status = f"Hovering over: {button.text}"
        else:
            self.hover_status = "No hover"
    
    def on_button_click(self, button):
        """Callback for click events"""
        self.click_count += 1
        self.click_status = f"Clicked: {button.text} ({self.click_count} times)"
        print(f"Button '{button.text}' was clicked!")
    
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        
        # Get mouse state
        mouse_pos = pygame.mouse.get_pos()
        mouse_pressed = pygame.mouse.get_pressed()[0]
        
        # Handle button events using the new comprehensive method
        self.start_button.handle_events(mouse_pos, mouse_pressed)
        self.stop_button.handle_events(mouse_pos, mouse_pressed)
        self.action_button.handle_events(mouse_pos, mouse_pressed)
        
        return True
    
    def draw(self):
        """Draw everything"""
        self.screen.fill(BLACK)
        
        # Draw title
        title_text = self.font.render("Enhanced Button Example", True, WHITE)
        self.screen.blit(title_text, (WINDOW_WIDTH // 2 - title_text.get_width() // 2, 20))
        
        # Draw buttons
        self.start_button.draw(self.screen)
        self.stop_button.draw(self.screen)
        self.action_button.draw(self.screen)
        
        # Draw status information
        status_font = pygame.font.Font(None, 24)
        
        hover_text = status_font.render(self.hover_status, True, WHITE)
        self.screen.blit(hover_text, (50, 200))
        
        click_text = status_font.render(self.click_status, True, WHITE)
        self.screen.blit(click_text, (50, 230))
        
        # Draw instructions
        instructions = [
            "Hover over buttons to see hover events",
            "Click buttons to see click events",
            "Events are handled automatically by the Button class"
        ]
        
        for i, instruction in enumerate(instructions):
            inst_text = status_font.render(instruction, True, WHITE)
            self.screen.blit(inst_text, (50, 300 + i * 30))
        
        pygame.display.flip()
    
    def run(self):
        """Main application loop"""
        running = True
        
        while running:
            running = self.handle_events()
            self.draw()
            self.clock.tick(FPS)
        
        pygame.quit()

if __name__ == "__main__":
    app = ButtonExample()
    app.run()
