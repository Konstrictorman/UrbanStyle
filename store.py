import pygame
import numpy as np
import sys

# Initialize pygame
pygame.init()

# Grid dimensions
environment_rows = 7
environment_cols = 9

# Window settings
CELL_SIZE = 60
GRID_WIDTH = environment_cols * CELL_SIZE
GRID_HEIGHT = environment_rows * CELL_SIZE
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 800
FPS = 60

# Grid positioning (center the grid in the window)
GRID_OFFSET_X = (WINDOW_WIDTH - GRID_WIDTH) // 2
GRID_OFFSET_Y = 50  # Leave space at top for controls

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (41, 182, 246)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
ORANGE = (255, 128, 0)

# Color mapping for different cell types
COLORS = {
    -100: GRAY,      # Obstacles/Storage areas
    -1: WHITE,       # Aisles (walkable paths)
    99: GREEN,       # Goal/Packaging area
    100: ORANGE,  # Goal/Packaging area
    101: BLUE  # Goal/Packaging area
}

# Initialize rewards matrix - all cells start as obstacles (-100)
rewards = np.full((environment_rows, environment_cols), -100)

# Define aisle locations (walkable paths)
# Note: Rows 0 and 6, Columns 0 and 8 are all obstacles (borders)
aisles = {}
aisles[1] = [1, 2, 3, 4, 5, 6, 7]  # Row 1: columns 1-7 are aisles
aisles[2] = [1, 4, 7]                  # Row 2: only columns 1 and 7 are aisles
aisles[3] = [1, 2, 3, 4, 5, 6, 7]  # Row 3: columns 1-7 are aisles
aisles[4] = [1, 4, 7]                  # Row 4: only columns 1 and 7 are aisles
aisles[5] = [1, 2, 3, 4, 5, 6, 7]  # Row 5: columns 1-7 are aisles

# Set rewards for aisle locations
for row_index in range(environment_rows):
    if row_index in aisles:
        for col_index in aisles[row_index]:
            rewards[row_index, col_index] = -1

# Set entrance at position (1,0)
rewards[1, 0] = -1
rewards[5, 8] = -1

# Set goal location (packaging area) - example at top-right accessible area
rewards[1, 7] = 99
rewards[3, 7] = 100
rewards[5, 2] = 101

def draw_grid(screen):
    """Draw the grid with different colors based on cell values"""
    for row in range(environment_rows):
        for col in range(environment_cols):
            # Calculate pixel position with offset
            x = col * CELL_SIZE + GRID_OFFSET_X
            y = row * CELL_SIZE + GRID_OFFSET_Y
            
            # Get color based on reward value
            reward_value = rewards[row, col]
            color = COLORS.get(reward_value, BLACK)
            
            # Draw cell
            pygame.draw.rect(screen, color, (x, y, CELL_SIZE, CELL_SIZE))
            
            # Draw grid lines
            pygame.draw.rect(screen, BLACK, (x, y, CELL_SIZE, CELL_SIZE), 2)
            
            # Add text to show coordinates (optional)
            font = pygame.font.Font(None, 24)
            text = font.render(f"{row},{col}", True, BLACK)
            text_rect = text.get_rect(center=(x + CELL_SIZE//2, y + CELL_SIZE//2))
            screen.blit(text, text_rect)

def draw_ui(screen):
    """Draw UI elements in the extra space"""
    font = pygame.font.Font(None, 36)
    title = font.render("Store Path Planning - 7x9 Grid", True, BLACK)
    screen.blit(title, (10, 10))
    
    # Add some info text
    font_small = pygame.font.Font(None, 24)
    info_text = [
        "Gray: Obstacles/Storage",
        "White: Aisles", 
        "Green: Beauty",
        "Orange: Electronics",
        "Blue: Clothing",
        "Entrance: (1,0)"
    ]
    
    for i, text in enumerate(info_text):
        rendered_text = font_small.render(text, True, BLACK)
        screen.blit(rendered_text, (10, GRID_OFFSET_Y + GRID_HEIGHT + 20 + i * 25))

def main():
    """Main game loop"""
    # Create display window
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Store Path Planning")
    clock = pygame.time.Clock()
    
    # Print rewards matrix to console
    print("Rewards Matrix:")
    for row in rewards:
        print(row)
    print()
    
    # Main game loop
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Clear screen
        screen.fill(LIGHT_GRAY)
        
        # Draw UI elements
        draw_ui(screen)
        
        # Draw grid
        draw_grid(screen)
        
        # Update display
        pygame.display.flip()
        clock.tick(FPS)
    
    # Quit
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
