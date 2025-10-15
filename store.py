import pygame
import numpy as np
import sys
import threading
import time
import os
from pygame import gfxdraw
from concurrent.futures import ThreadPoolExecutor
import queue

# Import UI components
from button import Button
from progressbar import ProgressBar
from inputfield import InputField

# Initialize pygame
pygame.init()

# Grid dimensions
environment_rows = 7
environment_cols = 9

# Window settings
CELL_SIZE = 60
GRID_WIDTH = environment_cols * CELL_SIZE
GRID_HEIGHT = environment_rows * CELL_SIZE
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 800
FPS = 60

# Control panel positioning (left side)
CONTROL_PANEL_WIDTH = 400  # Increased from 350 to 400 (50px wider)
CONTROL_PANEL_HEIGHT = WINDOW_HEIGHT - 40
CONTROL_PANEL_X = 20
CONTROL_PANEL_Y = 20

# Grid positioning (top-right area)
GRID_OFFSET_X = CONTROL_PANEL_X + CONTROL_PANEL_WIDTH + 20  # 20px gap from control panel
GRID_OFFSET_Y = 50  # 50px from top

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (41, 182, 246)
YELLOW = (255, 255, 0)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
ORANGE = (255, 128, 0)
PURPLE = (128, 0, 128)
CYAN = (0, 255, 255)
DARK_GRAY = (64, 64, 64)

# Color mapping for different cell types
COLORS = {
    -1: WHITE,       # Aisles (walkable paths)
    -100: GRAY,      # Obstacles/Storage areas
    95: YELLOW,      # Entrances
    99: GREEN,       # Goal 1
    100: ORANGE,     # Goal 2
    101: BLUE,       # Goal 3
    'start': PURPLE, # Start position
    'end': RED,      # End position
    'path': CYAN     # Path highlighting
}

# Initialize rewards matrix - all cells start as obstacles (-100)
rewards = np.full((environment_rows, environment_cols), -100)

# Define aisle locations (walkable paths)
aisles = {}
aisles[1] = [1, 2, 3, 4, 5, 6, 7]  # Row 1: columns 1-7 are aisles
aisles[2] = [1, 4, 7]              # Row 2: only columns 1 and 7 are aisles
aisles[3] = [1, 2, 3, 4, 5, 6, 7]  # Row 3: columns 1-7 are aisles
aisles[4] = [1, 4, 7]              # Row 4: only columns 1 and 7 are aisles
aisles[5] = [1, 2, 3, 4, 5, 6, 7]  # Row 5: columns 1-7 are aisles

# Set rewards for aisle locations
for row_index in range(environment_rows):
    if row_index in aisles:
        for col_index in aisles[row_index]:
            rewards[row_index, col_index] = -1

# Define all goals (entrances and store sections)
goals = {
    (1, 0): {"name": "Entrance 1", "reward": 100, "color": YELLOW},
    (5, 8): {"name": "Entrance 2", "reward": 100, "color": YELLOW},
    (3, 7): {"name": "Electronics", "reward": 100, "color": ORANGE},
    (1, 3): {"name": "Beauty", "reward": 100, "color": GREEN},
    (5, 1): {"name": "Clothing", "reward": 100, "color": BLUE}
}

# Set entrance and goal locations with rewards
for (row, col), info in goals.items():
    rewards[row, col] = info["reward"]

# Q-Learning parameters
actions = ['up', 'right', 'down', 'left']
epsilon = 0.9
discount_factor = 0.9
learning_rate = 0.9

# Separate Q-tables for each goal
q_tables = {}
for goal_pos, goal_info in goals.items():
    q_tables[goal_pos] = np.zeros((environment_rows, environment_cols, 4))

# Training state
training = False
training_progress = 0
max_episodes = 3000
current_episode = 0

# Thread-safe progress tracking for parallel training
progress_lock = threading.Lock()
goal_progress = {}  # Track progress for each goal
completed_goals = 0
total_goals = 0

# Pathfinding state
start_position = None
end_position = None
current_path = []
pathfinding_mode = False
selected_goal = None

# UI button instances (for persistent hover effects)
goal_buttons = []

# Training parameters (user adjustable)
user_epsilon = 0.9
user_discount_factor = 0.9
user_learning_rate = 0.9

# Simple black circles will be used for path visualization
# No need for complex image loading or rotation logic

# Input field class moved to inputfield.py

# Button class moved to button.py

# Progress bar class moved to progressbar.py

# Q-Learning functions (integrated from storeQLearning.py)
def is_terminal_state(current_row_index, current_col_index):
    if rewards[current_row_index, current_col_index] == -1.:
        return False
    else:
        return True

def get_starting_location():
    current_row_index = np.random.randint(environment_rows)
    current_col_index = np.random.randint(environment_cols)
    while is_terminal_state(current_row_index, current_col_index):
        current_row_index = np.random.randint(environment_rows)
        current_col_index = np.random.randint(environment_cols)
    return current_row_index, current_col_index

def get_next_action(current_row_index, current_col_index, epsilon, q_values):
    if np.random.random() < epsilon:
        return np.argmax(q_values[current_row_index, current_col_index])
    else:
        return np.random.randint(4)

def get_next_location(current_row_index, current_col_index, action_index):
    new_row_index = current_row_index
    new_col_index = current_col_index
    if actions[action_index] == 'up' and current_row_index > 0:
        new_row_index -= 1
    elif actions[action_index] == 'right' and current_col_index < environment_cols-1:
        new_col_index += 1
    elif actions[action_index] == 'down' and current_row_index < environment_rows-1:
        new_row_index += 1
    elif actions[action_index] == 'left' and current_col_index > 0:
        new_col_index -= 1
    return new_row_index, new_col_index

def train_single_goal(target_goal, goal_idx, total_goals):
    """Train a single Q-table for a specific goal (thread-safe)"""
    global training, q_tables, goal_progress, completed_goals, user_epsilon, user_discount_factor, user_learning_rate
    
    goal_name = goals[target_goal]["name"]
    print(f"Training Q-table for {goal_name} at {target_goal}...")
    
    # Get the specific Q-table for this goal
    q_values = q_tables[target_goal]
    local_epsilon = user_epsilon  # Use user-adjustable epsilon
    local_discount_factor = user_discount_factor  # Use user-adjustable discount factor
    local_learning_rate = user_learning_rate  # Use user-adjustable learning rate
    
    # Initialize progress for this goal
    with progress_lock:
        goal_progress[target_goal] = 0
    
    # Train this specific Q-table
    for episode in range(max_episodes):
        if not training:  # Check if training was stopped
            break
            
        # Get the starting location for this episode (not at target goal)
        row_index, col_index = get_starting_location()
        while (row_index, col_index) == target_goal:
            row_index, col_index = get_starting_location()

        # Continue taking actions until we reach the target goal
        max_steps_per_episode = 50
        step_count = 0
        
        while (row_index, col_index) != target_goal and step_count < max_steps_per_episode:
            # Choose which action to take
            action_index = get_next_action(row_index, col_index, local_epsilon, q_values)

            # Perform the chosen action and transition to the next state
            old_row, old_col = row_index, col_index
            new_row, new_col = get_next_location(row_index, col_index, action_index)
            
            # Only update if the move is valid (not hitting walls)
            if rewards[new_row, new_col] != -100:
                row_index, col_index = new_row, new_col

            # Calculate reward - only this specific goal gets high reward
            if (row_index, col_index) == target_goal:
                reward = 100  # High reward for reaching THIS specific goal
            else:
                reward = -1   # Small penalty for each step
            
            old_q_value = q_values[old_row, old_col, action_index]
            
            # Calculate temporal difference
            if (row_index, col_index) == target_goal:
                # If we reached the target goal, no future reward
                temporal_difference = reward - old_q_value
            else:
                # If we're still moving, consider future rewards
                temporal_difference = reward + (local_discount_factor * np.max(q_values[row_index, col_index])) - old_q_value

            # Update the Q-Value for the previous state and action pair
            new_q_value = old_q_value + (local_learning_rate * temporal_difference)
            q_values[old_row, old_col, action_index] = new_q_value
            
            step_count += 1
        
        # Decay epsilon over time for better convergence
        if episode % 500 == 0 and episode > 0:
            local_epsilon = max(0.1, local_epsilon * 0.95)
        
        # Update progress for this goal (thread-safe)
        with progress_lock:
            goal_progress[target_goal] = (episode + 1) / max_episodes * 100
        
        # Small delay to allow UI updates
        time.sleep(0.001)
    
    # Mark this goal as completed
    with progress_lock:
        completed_goals += 1
        print(f"✓ Completed training for {goal_name} ({completed_goals}/{total_goals})")

def train_q_learning_parallel():
    """Train all Q-tables in parallel using threading"""
    global training, training_progress, current_episode, total_goals, goal_progress, completed_goals
    
    training = True
    training_progress = 0
    current_episode = 0
    
    # Initialize progress tracking
    goal_positions = list(goals.keys())
    total_goals = len(goal_positions)
    completed_goals = 0
    goal_progress = {}
    
    print(f"Starting parallel training for {total_goals} goals...")
    print("Each goal will train for 1000 episodes simultaneously.")
    
    # Create and start threads for each goal
    threads = []
    for goal_idx, target_goal in enumerate(goal_positions):
        thread = threading.Thread(
            target=train_single_goal,
            args=(target_goal, goal_idx, total_goals),
            daemon=True
        )
        threads.append(thread)
        thread.start()
    
    # Monitor progress until all threads complete
    while completed_goals < total_goals and training:
        # Calculate overall progress
        with progress_lock:
            if goal_progress:
                overall_progress = sum(goal_progress.values()) / len(goal_progress)
                training_progress = overall_progress
        
        time.sleep(0.1)  # Update progress every 100ms
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join(timeout=1)
    
    training = False
    print("Training complete! All 5 Q-tables trained in parallel:")
    for pos, info in goals.items():
        print(f"  - {info['name']} at {pos}")
    
    # Enable pathfinding buttons after training completes
    enable_pathfinding_buttons()

def train_q_learning():
    """Wrapper function to choose between parallel and sequential training"""
    # For now, use parallel training
    train_q_learning_parallel()

def find_path(start_row, start_col, end_row, end_col):
    global current_path
    
    # Use the appropriate Q-table based on the destination
    target_goal = (end_row, end_col)
    if target_goal not in q_tables:
        print(f"No Q-table found for goal at {target_goal}!")
        return []
    
    q_values = q_tables[target_goal]
    
    if q_values.sum() == 0:
        print(f"Q-table for {goals[target_goal]['name']} not trained yet!")
        return []
    
    current_row, current_col = start_row, start_col
    path = [(current_row, current_col)]
    visited = set()
    visited.add((current_row, current_col))
    
    max_steps = 30
    step_count = 0
    
    # Continue until we reach the specific goal
    while (current_row != end_row or current_col != end_col) and step_count < max_steps:
        # Get the best action to take (exploit learned policy)
        action_index = np.argmax(q_values[current_row, current_col])
        
        # Try to move in the direction of the best action
        new_row, new_col = get_next_location(current_row, current_col, action_index)
        
        # If the best action leads to a visited location or invalid location, try other actions
        if (new_row, new_col) in visited or rewards[new_row, new_col] == -100:
            # Try all actions and pick the best valid one
            best_action = -1
            best_value = float('-inf')
            for action in range(4):
                test_row, test_col = get_next_location(current_row, current_col, action)
                if (test_row, test_col) not in visited and rewards[test_row, test_col] != -100:
                    if q_values[current_row, current_col, action] > best_value:
                        best_value = q_values[current_row, current_col, action]
                        best_action = action
            
            if best_action != -1:
                new_row, new_col = get_next_location(current_row, current_col, best_action)
            else:
                break  # No valid moves available
        
        # Move to the next location on the path
        current_row, current_col = new_row, new_col
        path.append((current_row, current_col))
        visited.add((current_row, current_col))
        step_count += 1
    
    current_path = path
    return path

def draw_grid(screen):
    """Draw the grid with different colors based on cell values"""
    for row in range(environment_rows):
        for col in range(environment_cols):
            # Calculate pixel position with offset
            x = col * CELL_SIZE + GRID_OFFSET_X
            y = row * CELL_SIZE + GRID_OFFSET_Y
            
            # Get color based on goal or reward value
            if (row, col) in goals:
                color = goals[(row, col)]["color"]
            else:
                reward_value = rewards[row, col]
                color = COLORS.get(reward_value, BLACK)
            
            # Highlight start position
            if start_position and (row, col) == start_position:
                color = COLORS['start']
            
            # Highlight end position
            if end_position and (row, col) == end_position:
                color = COLORS['end']
            
            # Draw cell
            pygame.draw.rect(screen, color, (x, y, CELL_SIZE, CELL_SIZE))
            
            # Draw grid lines
            pygame.draw.rect(screen, BLACK, (x, y, CELL_SIZE, CELL_SIZE), 2)
            
            # Add text to show coordinates (small font, bottom-left position)
            font = pygame.font.Font(None, 14)
            text = font.render(f"{row},{col}", True, BLACK)
            # Position in bottom-left corner with padding
            padding = 4
            text_x = x + padding
            text_y = y + CELL_SIZE - text.get_height() - padding
            screen.blit(text, (text_x, text_y))

    # Draw black circles for the path (after drawing the grid)
    if current_path:
        for pos in current_path:
            if pos != start_position and pos != end_position:  # Don't overlap with start/end markers
                row, col = pos
                x = col * CELL_SIZE + GRID_OFFSET_X
                y = row * CELL_SIZE + GRID_OFFSET_Y
                
                # Calculate center position for the circle
                center_x = x + CELL_SIZE // 2
                center_y = y + CELL_SIZE // 2
                
                # Draw a small black circle to show the path
                circle_radius = 8  # Small circle radius
                pygame.draw.circle(screen, BLACK, (center_x, center_y), circle_radius)

def get_goal_buttons():
    """Get list of goal buttons for click detection"""
    return goal_buttons

def create_goal_buttons():
    """Create persistent goal buttons"""
    global goal_buttons
    goal_buttons = []
    y_offset = CONTROL_PANEL_Y + 430  # Match the rendering position
    
    # Calculate button layout with more space
    button_width = 180  # Increased from 160 to 180
    button_spacing = 200  # Increased from 170 to 200
    buttons_per_row = 2
    
    for i, ((row, col), info) in enumerate(goals.items()):
        button_x = CONTROL_PANEL_X + 10 + (i % buttons_per_row) * button_spacing
        button_y = y_offset + (i // buttons_per_row) * 35
        
        # Create button with goal color and auto-contrasting text
        button = Button(button_x, button_y, button_width, 30, 
                       info["name"], info["color"], enabled=False)  # Start disabled
        goal_buttons.append(((row, col), button))
    return goal_buttons

def enable_pathfinding_buttons():
    """Enable pathfinding buttons after training completes"""
    global set_start_button, clear_path_button, goal_buttons
    
    # Enable main pathfinding buttons (goal buttons will be enabled when start position is set)
    set_start_button.set_enabled(True)
    clear_path_button.set_enabled(True)
    
    # Goal buttons remain disabled until start position is set
    print("Pathfinding buttons enabled!")

def enable_goal_buttons():
    """Enable goal buttons when start position is set"""
    global goal_buttons
    
    # Enable goal buttons
    for (goal_pos, button) in goal_buttons:
        button.set_enabled(True)
    
    print("Goal buttons enabled!")

def disable_goal_buttons():
    """Disable goal buttons when start position is cleared"""
    global goal_buttons
    
    # Disable goal buttons
    for (goal_pos, button) in goal_buttons:
        button.set_enabled(False)
    
    print("Goal buttons disabled!")

def draw_control_panel(screen):
    """Draw the control panel on the left side"""
    # Control panel background
    panel_rect = pygame.Rect(CONTROL_PANEL_X, CONTROL_PANEL_Y, CONTROL_PANEL_WIDTH, CONTROL_PANEL_HEIGHT)
    pygame.draw.rect(screen, LIGHT_GRAY, panel_rect)
    pygame.draw.rect(screen, WHITE, panel_rect, 3)
    
    # Title
    font_large = pygame.font.Font(None, 36)
    title = font_large.render("Store Path Planning", True, BLACK)
    screen.blit(title, (CONTROL_PANEL_X + 10, CONTROL_PANEL_Y + 10))
    
    # Training parameters section
    font_medium = pygame.font.Font(None, 28)
    params_title = font_medium.render("Training Parameters", True, BLACK)
    screen.blit(params_title, (CONTROL_PANEL_X + 10, CONTROL_PANEL_Y + 60))
    
    # Draw parameter input fields (positioned below the title with proper spacing)
    epsilon_field.draw(screen)
    discount_field.draw(screen)
    learning_field.draw(screen)
    
    # Training button
    train_button.draw(screen)
    
    # Progress bar
    progress_bar.draw(screen)
    
    # Progress text
    font_small = pygame.font.Font(None, 20)
    if training and goal_progress:
        # Show parallel training progress
        with progress_lock:
            completed_count = completed_goals
            progress_text = f"Parallel Training: {completed_count}/{total_goals} goals complete ({training_progress:.1f}%)"
    elif not training and any(q_table.sum() > 0 for q_table in q_tables.values()):
        # Training completed - show 100%
        progress_text = f"Training Complete: 5/5 goals complete (100.0%)"
    else:
        # Show sequential training progress
        total_episodes = len(goals) * max_episodes
        progress_text = f"Progress: {current_episode}/{total_episodes} episodes ({training_progress:.1f}%)"
    
    screen.blit(font_small.render(progress_text, True, BLACK), 
                (CONTROL_PANEL_X + 10, CONTROL_PANEL_Y + 225))
    
    # Training status - check if any Q-table is trained
    any_trained = any(q_table.sum() > 0 for q_table in q_tables.values())
    status_text = "Training..." if training else "Ready" if any_trained else "Not trained"
    status_color = ORANGE if training else GREEN if any_trained else RED
    screen.blit(font_small.render(f"Status: {status_text}", True, status_color), 
                (CONTROL_PANEL_X + 10, CONTROL_PANEL_Y + 245))
    
    
    # Pathfinding section
    pathfinding_title = font_medium.render("Pathfinding", True, BLACK)
    screen.blit(pathfinding_title, (CONTROL_PANEL_X + 10, CONTROL_PANEL_Y + 280))
    
    # Instructions
    instructions = [
        "1. Click 'Set Start' then click on grid",
        "2. Select a destination goal below", 
        "3. Click 'Find Path' to get optimal route",
        "4. Click 'Clear Path' to reset"
    ]
    
    for i, instruction in enumerate(instructions):
        screen.blit(font_small.render(instruction, True, BLACK), 
                    (CONTROL_PANEL_X + 10, CONTROL_PANEL_Y + 310 + i * 20))
    
    # Goal selection section
    goal_title = font_medium.render("Select Destination:", True, BLACK)
    screen.blit(goal_title, (CONTROL_PANEL_X + 10, CONTROL_PANEL_Y + 400))
    
    # Draw goal buttons (using persistent buttons)
    for (goal_pos, button) in goal_buttons:
        button.draw(screen)
    
    # Pathfinding buttons
    set_start_button.draw(screen)
    clear_path_button.draw(screen)
    
    # Current positions display (below the buttons)
    if start_position:
        start_text = f"Start: ({start_position[0]}, {start_position[1]})"
        screen.blit(font_small.render(start_text, True, BLACK), 
                    (CONTROL_PANEL_X + 10, CONTROL_PANEL_Y + 590))
    
    if selected_goal:
        goal_name = goals[selected_goal]["name"]
        goal_text = f"Destination: {goal_name} ({selected_goal[0]}, {selected_goal[1]})"
        screen.blit(font_small.render(goal_text, True, BLACK), 
                    (CONTROL_PANEL_X + 10, CONTROL_PANEL_Y + 610))

def get_grid_position(mouse_pos):
    """Convert mouse position to grid coordinates"""
    mouse_x, mouse_y = mouse_pos
    
    # Check if mouse is within grid bounds
    if (GRID_OFFSET_X <= mouse_x <= GRID_OFFSET_X + GRID_WIDTH and
        GRID_OFFSET_Y <= mouse_y <= GRID_OFFSET_Y + GRID_HEIGHT):
        
        col = (mouse_x - GRID_OFFSET_X) // CELL_SIZE
        row = (mouse_y - GRID_OFFSET_Y) // CELL_SIZE
        
        # Check if it's a valid position (not an obstacle)
        if 0 <= row < environment_rows and 0 <= col < environment_cols:
            if rewards[row, col] != -100:  # Not an obstacle
                return row, col
    
    return None

def main():
    """Main game loop"""
    global training, pathfinding_mode, start_position, end_position, current_path, selected_goal
    
    # Create display window
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Store Path Planning with Q-Learning")
    clock = pygame.time.Clock()
    
    # Initialize UI components
    global train_button, progress_bar, set_start_button, clear_path_button
    global epsilon_field, discount_field, learning_field
    global user_epsilon, user_discount_factor, user_learning_rate
    
    # Initialize parameter input fields
    epsilon_field = InputField(CONTROL_PANEL_X + 10, CONTROL_PANEL_Y + 100, 100, 25, "Epsilon:", user_epsilon)
    discount_field = InputField(CONTROL_PANEL_X + 120, CONTROL_PANEL_Y + 100, 100, 25, "Discount:", user_discount_factor)
    learning_field = InputField(CONTROL_PANEL_X + 230, CONTROL_PANEL_Y + 100, 100, 25, "Learning:", user_learning_rate)
    
    train_button = Button(CONTROL_PANEL_X + 10, CONTROL_PANEL_Y + 140, 150, 40, 
                         "Start Training", GREEN if not training else GRAY, enabled=True)
    
    progress_bar = ProgressBar(CONTROL_PANEL_X + 10, CONTROL_PANEL_Y + 190, 370, 30, 100)  # Increased from 320 to 370
    
    set_start_button = Button(CONTROL_PANEL_X + 10, CONTROL_PANEL_Y + 540, 180, 35, 
                             "Set Start", BLUE, enabled=False)  # Start disabled, wider button
    
    clear_path_button = Button(CONTROL_PANEL_X + 200, CONTROL_PANEL_Y + 540, 180, 35, 
                              "Clear Path", RED, enabled=False)  # Start disabled, wider button
    
    # Create goal buttons
    create_goal_buttons()
    
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
                
                # Handle parameter input field keyboard input
                if epsilon_field.handle_input(event):
                    user_epsilon = epsilon_field.value
                if discount_field.handle_input(event):
                    user_discount_factor = discount_field.value
                if learning_field.handle_input(event):
                    user_learning_rate = learning_field.value
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    mouse_pos = pygame.mouse.get_pos()
                    
                    # Handle parameter input field clicks
                    epsilon_field.handle_click(mouse_pos)
                    discount_field.handle_click(mouse_pos)
                    learning_field.handle_click(mouse_pos)
                    
                    # Check button clicks
                    if train_button.is_clicked(mouse_pos) and not training:
                        # Start training in a separate thread
                        training_thread = threading.Thread(target=train_q_learning)
                        training_thread.daemon = True
                        training_thread.start()
                    
                    elif set_start_button.is_clicked(mouse_pos):
                        pathfinding_mode = "start"
                        current_path = []  # Clear current path
                    
                    elif clear_path_button.is_clicked(mouse_pos):
                        start_position = None
                        selected_goal = None
                        current_path = []
                        pathfinding_mode = False
                        disable_goal_buttons()  # Disable goal buttons when clearing
                    
                    # Check goal button clicks
                    else:
                        goal_clicked = False
                        for (goal_pos, goal_button) in get_goal_buttons():
                            if goal_button.is_clicked(mouse_pos):
                                # Check if start position is set
                                if not start_position:
                                    print("Please set start position first!")
                                    goal_clicked = True
                                    break
                                
                                selected_goal = goal_pos
                                goal_name = goals[goal_pos]["name"]
                                print(f"Destination selected: {goal_name} at {goal_pos}")
                                
                                # Check if the specific Q-table for this goal is trained
                                if q_tables[selected_goal].sum() > 0:
                                    goal_row, goal_col = selected_goal
                                    path = find_path(start_position[0], start_position[1], 
                                                   goal_row, goal_col)
                                    if path:
                                        print(f"Path to {goal_name} found with {len(path)} steps: {path}")
                                    else:
                                        print("No path found!")
                                else:
                                    print(f"Q-table for {goal_name} not trained yet! Please train the model first.")
                                
                                goal_clicked = True
                                break
                        
                        if not goal_clicked:
                            # Handle grid clicks for position setting
                            if pathfinding_mode == "start":
                                grid_pos = get_grid_position(mouse_pos)
                                if grid_pos:
                                    start_position = grid_pos
                                    print(f"Start position set to: {grid_pos}")
                                    pathfinding_mode = False
                                    enable_goal_buttons()  # Enable goal buttons when start position is set
                    
        
        # Handle hover effects for all buttons
        mouse_pos = pygame.mouse.get_pos()
        mouse_pressed = pygame.mouse.get_pressed()[0]  # Left mouse button
        
        # Update hover states for main buttons
        train_button.update_hover(mouse_pos)
        set_start_button.update_hover(mouse_pos)
        clear_path_button.update_hover(mouse_pos)
        
        # Update hover states for goal buttons
        for (goal_pos, goal_button) in get_goal_buttons():
            goal_button.update_hover(mouse_pos)
        
        # Clear screen
        screen.fill(BLACK)
        
        # Update progress bar
        # Show 100% if training is complete, otherwise show actual progress
        if not training and any(q_table.sum() > 0 for q_table in q_tables.values()):
            progress_bar.update(100)  # Show 100% when training completes
        else:
            progress_bar.update(training_progress)
        
        # Draw control panel
        draw_control_panel(screen)
        
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