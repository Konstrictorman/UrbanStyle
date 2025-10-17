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
WINDOW_WIDTH = 1520  # Increased to 1850 for new right panel
WINDOW_HEIGHT = 800  # Increased to accommodate heat map below store grid
FPS = 60

# Control panel positioning (left side)
CONTROL_PANEL_WIDTH = 450  # Left control panel width
CONTROL_PANEL_HEIGHT = WINDOW_HEIGHT - 40
CONTROL_PANEL_X = 20
CONTROL_PANEL_Y = 20

# Grid positioning (center area)
GRID_OFFSET_X = CONTROL_PANEL_X + CONTROL_PANEL_WIDTH + 20  # 20px gap from left control panel
GRID_OFFSET_Y = 50  # 50px from top

# Right control panel positioning (for customer simulation)
RIGHT_PANEL_WIDTH = 450  # Right control panel width
RIGHT_PANEL_HEIGHT = WINDOW_HEIGHT - 40
RIGHT_PANEL_X = GRID_OFFSET_X + GRID_WIDTH + 20  # 20px gap from grid
RIGHT_PANEL_Y = 20

# Heat map positioning (below the main store grid)
HEATMAP_Y = GRID_OFFSET_Y + GRID_HEIGHT + 20
HEATMAP_HEIGHT = 290

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

# Customer simulation system
class Customer:
    def __init__(self, customer_id):
        self.id = customer_id
        self.current_position = None
        self.target_departments = []
        self.visited_departments = []
        self.current_path = []
        self.path_index = 0
        self.entrance = None
        self.exit = None
        self.state = "entering"  # entering, shopping, exiting, finished
        self.shopping_time = 0
        self.total_time = 0
        self.departments_to_visit = 0
        
    def generate_shopping_plan(self):
        """Generate a random shopping plan for this customer"""
        # Randomly choose 1-3 departments to visit
        self.departments_to_visit = np.random.randint(1, 4)
        
        # Available departments (excluding entrances)
        departments = [(3, 7), (1, 3), (5, 1)]  # Electronics, Beauty, Clothing
        department_names = ["Electronics", "Beauty", "Clothing"]
        
        # Randomly select departments to visit
        selected_indices = np.random.choice(len(departments), self.departments_to_visit, replace=False)
        self.target_departments = [departments[i] for i in selected_indices]
        
        # Choose random entrance
        entrances = [(1, 0), (5, 8)]
        self.entrance = entrances[np.random.randint(0, len(entrances))]
        
        # Choose closest exit to final department
        exits = [(1, 0), (5, 8)]
        final_dept = self.target_departments[-1]
        # Calculate distance to each exit and choose closest
        distances = []
        for exit_pos in exits:
            dist = abs(final_dept[0] - exit_pos[0]) + abs(final_dept[1] - exit_pos[1])
            distances.append(dist)
        self.exit = exits[np.argmin(distances)]
        
        self.current_position = self.entrance
        self.state = "entering"
        
    def get_next_target(self):
        """Get the next department to visit"""
        if len(self.visited_departments) < len(self.target_departments):
            return self.target_departments[len(self.visited_departments)]
        else:
            return self.exit
    
    def reached_target(self):
        """Check if customer has reached their current target"""
        if not self.current_position or not self.current_path:
            return False
        
        target = self.get_next_target()
        return self.current_position == target
    
    def move_to_next_position(self):
        """Move customer to next position in their path"""
        if self.path_index < len(self.current_path):
            self.current_position = self.current_path[self.path_index]
            self.path_index += 1
            return True
        return False
    
    def complete_department_visit(self):
        """Mark current department as visited"""
        if self.current_position in self.target_departments:
            self.visited_departments.append(self.current_position)
            self.shopping_time += np.random.randint(5, 15)  # 5-15 minutes shopping time
            
        # Check if done shopping
        if len(self.visited_departments) == len(self.target_departments):
            self.state = "exiting"
        else:
            self.state = "shopping"

# Heat map data structure to track customer visits per grid cell
heat_map = np.zeros((environment_rows, environment_cols), dtype=int)

# Customer simulation state
customer_simulation = {
    'running': False,
    'customers': [],
    'total_customers': 10,
    'current_time': 0.0,
    'customer_id_counter': 1,
    'arrival_rate': 0.1,  # customers per minute
    'next_arrival_time': 0.0,
    'completed_customers': 0,
    'show_heatmap': True,
    'analytics': {
        'total_customers_served': 0,
        'total_shopping_time': 0,
        'department_visits': {'Electronics': 0, 'Beauty': 0, 'Clothing': 0},
        'entrance_usage': {'Entrance 1': 0, 'Entrance 2': 0},
        'exit_usage': {'Exit 1': 0, 'Exit 2': 0},
        'average_departments_per_customer': 0,
        'average_shopping_time': 0
    }
}

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

# Customer simulation functions
def start_customer_simulation():
    """Start the customer simulation"""
    global customer_simulation
    
    if not any(q_table.sum() > 0 for q_table in q_tables.values()):
        print("Error: Q-tables not trained yet! Please train the model first.")
        return
    
    # Update simulation parameters from input fields
    try:
        customer_simulation['total_customers'] = int(customer_count_field.value)
    except (ValueError, TypeError):
        customer_simulation['total_customers'] = 1000  # Fallback to default
    
    try:
        customer_simulation['arrival_rate'] = float(arrival_rate_field.value)
    except (ValueError, TypeError):
        customer_simulation['arrival_rate'] = 0.1  # Fallback to default
    
    customer_simulation['running'] = True
    customer_simulation['customers'] = []
    customer_simulation['current_time'] = 0.0
    customer_simulation['customer_id_counter'] = 1
    customer_simulation['completed_customers'] = 0
    customer_simulation['next_arrival_time'] = np.random.exponential(1.0 / customer_simulation['arrival_rate'])
    
    # Reset heat map for new simulation
    reset_heatmap()
    
    # Reset analytics
    customer_simulation['analytics'] = {
        'total_customers_served': 0,
        'total_shopping_time': 0,
        'department_visits': {'Electronics': 0, 'Beauty': 0, 'Clothing': 0},
        'entrance_usage': {'Entrance 1': 0, 'Entrance 2': 0},
        'exit_usage': {'Exit 1': 0, 'Exit 2': 0},
        'average_departments_per_customer': 0,
        'average_shopping_time': 0
    }
    
    print(f"Starting customer simulation with {customer_simulation['total_customers']} customers at rate {customer_simulation['arrival_rate']} customers/minute...")

def stop_customer_simulation():
    """Stop the customer simulation"""
    global customer_simulation
    customer_simulation['running'] = False
    print("Customer simulation stopped.")

def update_customer_simulation():
    """Update customer simulation state"""
    global customer_simulation
    
    if not customer_simulation['running']:
        return
    
    # Use the customer count that was set when simulation started
    total_customers = customer_simulation['total_customers']
    
    # Check if we've generated all customers and all are finished
    if customer_simulation['completed_customers'] >= total_customers and not customer_simulation['customers']:
        customer_simulation['running'] = False
        print(f"Simulation completed! All {total_customers} customers finished.")
        print_customer_analytics()
        # Re-enable start button and disable stop button when simulation completes
        # Note: We'll need to access the buttons from the main function
        return
    
    # Generate new customers based on arrival rate
    total_generated = customer_simulation['completed_customers'] + len(customer_simulation['customers'])
    
    # Advance time only if we still need to generate customers OR have active customers
    if total_generated < total_customers or len(customer_simulation['customers']) > 0:
        customer_simulation['current_time'] += 1
    
    if customer_simulation['current_time'] >= customer_simulation['next_arrival_time'] and total_generated < total_customers:
        # Create new customer
        customer = Customer(customer_simulation['customer_id_counter'])
        customer_simulation['customer_id_counter'] += 1
        customer.generate_shopping_plan()
        
        # Generate path to first target
        target = customer.get_next_target()
        path = find_path(customer.current_position[0], customer.current_position[1], 
                       target[0], target[1])
        if path:
            customer.current_path = path
            customer_simulation['customers'].append(customer)
            
            # Update entrance usage analytics
            entrance_name = "Entrance 1" if customer.entrance == (1, 0) else "Entrance 2"
            customer_simulation['analytics']['entrance_usage'][entrance_name] += 1
            
            # Schedule next arrival
            customer_simulation['next_arrival_time'] = customer_simulation['current_time'] + np.random.exponential(1.0 / customer_simulation['arrival_rate'])
        else:
            print(f"Warning: Could not find path for customer {customer.id}")
    
    # Update existing customers
    customers_to_remove = []
    for customer in customer_simulation['customers']:
        customer.total_time += 1  # 1 minute per update
        
        
        if customer.state == "entering" or customer.state == "shopping" or customer.state == "exiting":
            # Move customer along path
            if customer.move_to_next_position():
                # Update heat map for customer movement
                if customer.current_position:
                    row, col = customer.current_position
                    heat_map[row, col] += 1
                # Check if reached target
                if customer.reached_target():
                    if customer.current_position in customer.target_departments:
                        # Reached a department
                        customer.complete_department_visit()
                        
                        # Update department visit analytics
                        dept_name = get_department_name(customer.current_position)
                        customer_simulation['analytics']['department_visits'][dept_name] += 1
                        
                        # Generate path to next target
                        target = customer.get_next_target()
                        path = find_path(customer.current_position[0], customer.current_position[1], 
                                       target[0], target[1])
                        if path:
                            customer.current_path = path
                            customer.path_index = 1  # Start from second step since first is current position
                        else:
                            print(f"Warning: Could not find path for customer {customer.id}")
                    else:
                        # Reached exit
                        customer.state = "finished"
                        customers_to_remove.append(customer)
                        
                        # Update exit usage analytics
                        exit_name = "Exit 1" if customer.exit == (1, 0) else "Exit 2"
                        customer_simulation['analytics']['exit_usage'][exit_name] += 1
                        
                        # Update analytics
                        customer_simulation['analytics']['total_customers_served'] += 1
                        customer_simulation['analytics']['total_shopping_time'] += customer.shopping_time
                        customer_simulation['completed_customers'] += 1
                        
                        # Track average departments per customer
                        total_departments_visited = len(customer.visited_departments)
                        current_avg = customer_simulation['analytics']['average_departments_per_customer']
                        total_customers = customer_simulation['analytics']['total_customers_served']
                        customer_simulation['analytics']['average_departments_per_customer'] = (
                            (current_avg * (total_customers - 1) + total_departments_visited) / total_customers
                        )
                        
                        # Update average shopping time
                        total_shopping_time = customer_simulation['analytics']['total_shopping_time']
                        customer_simulation['analytics']['average_shopping_time'] = total_shopping_time / total_customers
                        
                        if customer_simulation['completed_customers'] % 100 == 0:
                            print(f"Completed {customer_simulation['completed_customers']}/{total_customers} customers")
    
    # Remove finished customers
    for customer in customers_to_remove:
        customer_simulation['customers'].remove(customer)

def get_department_name(position):
    """Get department name from position"""
    dept_map = {
        (3, 7): "Electronics",
        (1, 3): "Beauty", 
        (5, 1): "Clothing"
    }
    return dept_map.get(position, "Unknown")

def get_heatmap_color(value, max_value):
    """Get color for heat map cell based on visit count"""
    if max_value == 0:
        return (240, 240, 240)  # Light gray for no visits
    
    # Normalize value to 0-1 range
    intensity = min(value / max_value, 1.0)
    
    # Create color gradient from light blue to dark red
    if intensity == 0:
        return (240, 240, 240)  # Light gray
    elif intensity < 0.2:
        return (173, 216, 230)  # Light blue
    elif intensity < 0.4:
        return (135, 206, 235)  # Sky blue
    elif intensity < 0.6:
        return (255, 255, 0)    # Yellow
    elif intensity < 0.8:
        return (255, 165, 0)    # Orange
    else:
        return (255, 0, 0)      # Red

def draw_heatmap(screen):
    """Draw the heat map below the store grid"""
    if not customer_simulation['show_heatmap']:
        return
    
    # Heat map background (width matches GRID_WIDTH)
    heatmap_width = GRID_WIDTH
    heatmap_rect = pygame.Rect(GRID_OFFSET_X, HEATMAP_Y, heatmap_width, HEATMAP_HEIGHT)
    pygame.draw.rect(screen, LIGHT_GRAY, heatmap_rect)
    pygame.draw.rect(screen, WHITE, heatmap_rect, 2)
    
    # Heat map title
    font_medium = pygame.font.Font(None, 24)
    title_text = font_medium.render("Customer Movement Heat Map", True, BLACK)
    screen.blit(title_text, (GRID_OFFSET_X + 10, HEATMAP_Y + 10))
    
    # Get maximum visit count for color scaling
    max_visits = np.max(heat_map)
    
    # Calculate cell size for heat map (reduced to fit smaller width)
    max_cell_width = heatmap_width // environment_cols
    heatmap_cell_size = min(max_cell_width - 2, 30)  # Further reduced to 30
    heatmap_start_x = GRID_OFFSET_X + 10
    heatmap_start_y = HEATMAP_Y + 40
    
    # Draw heat map cells
    for row in range(environment_rows):
        for col in range(environment_cols):
            cell_x = heatmap_start_x + col * heatmap_cell_size
            cell_y = heatmap_start_y + row * heatmap_cell_size
            
            # Skip walls (same as main grid)
            if rewards[row, col] == -100:
                color = (100, 100, 100)  # Dark gray for walls
            else:
                visit_count = heat_map[row, col]
                color = get_heatmap_color(visit_count, max_visits)
            
            # Draw cell
            cell_rect = pygame.Rect(cell_x, cell_y, heatmap_cell_size - 1, heatmap_cell_size - 1)
            pygame.draw.rect(screen, color, cell_rect)
    
    # Draw color scale legend
    legend_x = heatmap_start_x + heatmap_width - 150
    legend_y = HEATMAP_Y + 40
    legend_width = 120
    legend_height = 20
    
    # Legend title
    legend_title = font_medium.render("Visit Intensity", True, BLACK)
    screen.blit(legend_title, (legend_x, legend_y - 25))
    
    # Color gradient bar
    for i in range(legend_width):
        intensity = i / legend_width
        color = get_heatmap_color(intensity, 1.0)
        pygame.draw.line(screen, color, (legend_x + i, legend_y), (legend_x + i, legend_y + legend_height))
    
    # Legend labels
    font_small = pygame.font.Font(None, 14)
    min_label = font_small.render("0", True, BLACK)
    max_label = font_small.render(f"{max_visits}", True, BLACK)
    screen.blit(min_label, (legend_x, legend_y + legend_height + 5))
    screen.blit(max_label, (legend_x + legend_width - 20, legend_y + legend_height + 5))

def reset_heatmap():
    """Reset the heat map data"""
    global heat_map
    heat_map.fill(0)

def print_customer_analytics():
    """Print customer simulation analytics"""
    analytics = customer_simulation['analytics']
    
    print("\n" + "="*60)
    print("CUSTOMER SIMULATION ANALYTICS")
    print("="*60)
    
    print(f"Total Customers Served: {analytics['total_customers_served']}")
    print(f"Average Shopping Time: {analytics['total_shopping_time'] / max(1, analytics['total_customers_served']):.2f} minutes")
    
    print("\nDepartment Visits:")
    for dept, visits in analytics['department_visits'].items():
        percentage = (visits / max(1, analytics['total_customers_served'])) * 100
        print(f"  {dept}: {visits} visits ({percentage:.1f}%)")
    
    print("\nEntrance Usage:")
    for entrance, usage in analytics['entrance_usage'].items():
        percentage = (usage / max(1, analytics['total_customers_served'])) * 100
        print(f"  {entrance}: {usage} customers ({percentage:.1f}%)")
    
    print("\nExit Usage:")
    for exit, usage in analytics['exit_usage'].items():
        percentage = (usage / max(1, analytics['total_customers_served'])) * 100
        print(f"  {exit}: {usage} customers ({percentage:.1f}%)")
    
    print("="*60)

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
    
    # Draw customers
    for customer in customer_simulation['customers']:
        if customer.current_position:
            x = customer.current_position[1] * CELL_SIZE + GRID_OFFSET_X
            y = customer.current_position[0] * CELL_SIZE + GRID_OFFSET_Y
            
            # Color customers based on their state
            if customer.state == "entering":
                color = GREEN
            elif customer.state == "shopping":
                color = BLUE
            elif customer.state == "exiting":
                color = ORANGE
            else:
                color = WHITE
            
            # Draw customer as a small circle
            pygame.draw.circle(screen, color, (x + CELL_SIZE//2, y + CELL_SIZE//2), 8)
            
            # Draw customer ID as text (small)
            font_tiny = pygame.font.Font(None, 16)
            text = font_tiny.render(str(customer.id), True, BLACK)
            text_rect = text.get_rect(center=(x + CELL_SIZE//2, y + CELL_SIZE//2))
            screen.blit(text, text_rect)

def draw_right_control_panel(screen):
    """Draw the right control panel for customer simulation"""
    # Right panel background
    panel_rect = pygame.Rect(RIGHT_PANEL_X, RIGHT_PANEL_Y, RIGHT_PANEL_WIDTH, RIGHT_PANEL_HEIGHT)
    pygame.draw.rect(screen, LIGHT_GRAY, panel_rect)
    pygame.draw.rect(screen, WHITE, panel_rect, 3)
    
    # Title
    font_large = pygame.font.Font(None, 36)
    title = font_large.render("Customer Simulation", True, BLACK)
    screen.blit(title, (RIGHT_PANEL_X + 10, RIGHT_PANEL_Y + 10))
    
    # Draw customer simulation input fields
    customer_count_field.draw(screen)
    arrival_rate_field.draw(screen)
    
    # Draw help text for input fields
    font_small = pygame.font.Font(None, 16)
    # help_text1 = "Click field, type number, press ENTER"
    # help_text2 = "ESC to cancel, click elsewhere to finish"
    # screen.blit(font_small.render(help_text1, True, BLACK), (RIGHT_PANEL_X + 10, RIGHT_PANEL_Y + 90))
    # screen.blit(font_small.render(help_text2, True, BLACK), (RIGHT_PANEL_X + 10, RIGHT_PANEL_Y + 105))
    
    # Draw customer simulation buttons
    start_sim_button.draw(screen)
    stop_sim_button.draw(screen)
    
    # Customer simulation status
    font_small = pygame.font.Font(None, 20)
    total_customers = customer_simulation['total_customers']
    
    if customer_simulation['running']:
        active_customers = len(customer_simulation['customers'])
        completed_customers = customer_simulation['completed_customers']
        sim_status_text = f"Simulation Running - {completed_customers}/{total_customers} "
        sim_status_color = GREEN
    else:
        completed_customers = customer_simulation['completed_customers']
        sim_status_text = f"Simulation Stopped - {completed_customers}/{total_customers} customers completed"
        sim_status_color = GRAY
    
    screen.blit(font_small.render(sim_status_text, True, sim_status_color), 
                (RIGHT_PANEL_X + 10, RIGHT_PANEL_Y + 150))

    # Show simulation time
    time_text = f"Simulation Time: {customer_simulation['current_time']:.0f} minutes"
    screen.blit(font_small.render(time_text, True, BLACK), 
                (RIGHT_PANEL_X + 10, RIGHT_PANEL_Y + 170))
    
    # Analytics section
    font_medium = pygame.font.Font(None, 28)
    analytics_title = font_medium.render("Simulation Results", True, BLACK)
    screen.blit(analytics_title, (RIGHT_PANEL_X + 10, RIGHT_PANEL_Y + 220))
    
    analytics = customer_simulation['analytics']
    
    if analytics['total_customers_served'] > 0:
        y_offset = 250
        
        # Overall statistics
        avg_shopping_time = analytics['total_shopping_time'] / analytics['total_customers_served']
        screen.blit(font_small.render(f"Total Customers: {analytics['total_customers_served']}", True, BLACK), 
                    (RIGHT_PANEL_X + 10, RIGHT_PANEL_Y + y_offset))
        y_offset += 20
        
        screen.blit(font_small.render(f"Avg Shopping Time: {avg_shopping_time:.1f} min", True, BLACK), 
                    (RIGHT_PANEL_X + 10, RIGHT_PANEL_Y + y_offset))
        y_offset += 20
        
        screen.blit(font_small.render(f"Total Shopping Time: {analytics['total_shopping_time']} min", True, BLACK), 
                    (RIGHT_PANEL_X + 10, RIGHT_PANEL_Y + y_offset))
        y_offset += 20
        
        screen.blit(font_small.render(f"Avg Departments/Customer: {analytics['average_departments_per_customer']:.1f}", True, BLACK), 
                    (RIGHT_PANEL_X + 10, RIGHT_PANEL_Y + y_offset))
        y_offset += 30
        
        # Department visits
        screen.blit(font_small.render("Department Visits:", True, BLACK), 
                    (RIGHT_PANEL_X + 10, RIGHT_PANEL_Y + y_offset))
        y_offset += 20
        
        for dept, visits in analytics['department_visits'].items():
            percentage = (visits / analytics['total_customers_served']) * 100
            dept_text = f"  {dept}: {visits} ({percentage:.1f}%)"
            screen.blit(font_small.render(dept_text, True, BLACK), 
                        (RIGHT_PANEL_X + 20, RIGHT_PANEL_Y + y_offset))
            y_offset += 18
        
        y_offset += 10
        
        # Entrance usage
        screen.blit(font_small.render("Entrance Usage:", True, BLACK), 
                    (RIGHT_PANEL_X + 10, RIGHT_PANEL_Y + y_offset))
        y_offset += 20
        
        for entrance, usage in analytics['entrance_usage'].items():
            percentage = (usage / analytics['total_customers_served']) * 100
            entrance_text = f"  {entrance}: {usage} ({percentage:.1f}%)"
            screen.blit(font_small.render(entrance_text, True, BLACK), 
                        (RIGHT_PANEL_X + 20, RIGHT_PANEL_Y + y_offset))
            y_offset += 18
        
        y_offset += 10
        
        # Exit usage
        screen.blit(font_small.render("Exit Usage:", True, BLACK), 
                    (RIGHT_PANEL_X + 10, RIGHT_PANEL_Y + y_offset))
        y_offset += 20
        
        for exit, usage in analytics['exit_usage'].items():
            percentage = (usage / analytics['total_customers_served']) * 100
            exit_text = f"  {exit}: {usage} ({percentage:.1f}%)"
            screen.blit(font_small.render(exit_text, True, BLACK), 
                        (RIGHT_PANEL_X + 20, RIGHT_PANEL_Y + y_offset))
            y_offset += 18
    else:
        # Show message when no data available
        screen.blit(font_small.render("No simulation data yet", True, GRAY), 
                    (RIGHT_PANEL_X + 10, RIGHT_PANEL_Y + 250))
        screen.blit(font_small.render("Start a simulation to see results", True, GRAY), 
                    (RIGHT_PANEL_X + 10, RIGHT_PANEL_Y + 270))

def get_goal_buttons():
    """Get list of goal buttons for click detection"""
    return goal_buttons

def create_goal_buttons():
    """Create persistent goal buttons"""
    global goal_buttons
    goal_buttons = []
    y_offset = CONTROL_PANEL_Y + 450  # 20px down from original position (was 430)
    
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
    screen.blit(pathfinding_title, (CONTROL_PANEL_X + 10, CONTROL_PANEL_Y + 270))
    
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
    
    # Goal selection section (moved 20px down from original position)
    goal_title = font_medium.render("Select Destination:", True, BLACK)
    screen.blit(goal_title, (CONTROL_PANEL_X + 10, CONTROL_PANEL_Y + 420))
    
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
                    (CONTROL_PANEL_X + 10, CONTROL_PANEL_Y + 560))
    
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
    global start_sim_button, stop_sim_button, customer_count_field, arrival_rate_field
    global user_epsilon, user_discount_factor, user_learning_rate
    
    # Initialize parameter input fields
    epsilon_field = InputField(CONTROL_PANEL_X + 10, CONTROL_PANEL_Y + 100, 100, 25, "Epsilon:", user_epsilon)
    discount_field = InputField(CONTROL_PANEL_X + 120, CONTROL_PANEL_Y + 100, 100, 25, "Discount:", user_discount_factor)
    learning_field = InputField(CONTROL_PANEL_X + 230, CONTROL_PANEL_Y + 100, 100, 25, "Learning:", user_learning_rate)
    
    train_button = Button(CONTROL_PANEL_X + 10, CONTROL_PANEL_Y + 140, 150, 40, 
                         "Start Training", GREEN if not training else GRAY, enabled=True)
    
    progress_bar = ProgressBar(CONTROL_PANEL_X + 10, CONTROL_PANEL_Y + 190, 420, 30, 100)  # Increased width
    
    
    set_start_button = Button(CONTROL_PANEL_X + 10, CONTROL_PANEL_Y + 560, 180, 35, 
                             "Set Start", BLUE, enabled=False)  # Start disabled, wider button
    
    clear_path_button = Button(CONTROL_PANEL_X + 200, CONTROL_PANEL_Y + 560, 180, 35, 
                              "Clear Path", RED, enabled=False)  # Start disabled, wider button
    
    # Customer simulation controls (moved to right panel)
    customer_count_field = InputField(RIGHT_PANEL_X + 10, RIGHT_PANEL_Y + 60, 120, 25, "Customers:", 100, min_val=1, max_val=10000, is_integer=True)  # Allow 1-10000 customers
    arrival_rate_field = InputField(RIGHT_PANEL_X + 140, RIGHT_PANEL_Y + 60, 120, 25, "Rate:", 0.1, min_val=0.01, max_val=10.0, is_integer=False)  # Allow 0.01-10.0 rate
    
    start_sim_button = Button(RIGHT_PANEL_X + 10, RIGHT_PANEL_Y + 100, 120, 35, 
                             "Start Simulation", GREEN, enabled=True)
    
    stop_sim_button = Button(RIGHT_PANEL_X + 140, RIGHT_PANEL_Y + 100, 120, 35, 
                            "Stop Simulation", RED, enabled=False)
    
    # Heat map control buttons
    toggle_heatmap_button = Button(RIGHT_PANEL_X + 10, RIGHT_PANEL_Y + 145, 120, 35, 
                                  "Toggle Heat Map", BLUE, enabled=True)
    
    reset_heatmap_button = Button(RIGHT_PANEL_X + 140, RIGHT_PANEL_Y + 145, 120, 35, 
                                 "Reset Heat Map", ORANGE, enabled=True)
    
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
                if customer_count_field.handle_input(event):
                    customer_simulation['total_customers'] = int(customer_count_field.value)
                if arrival_rate_field.handle_input(event):
                    customer_simulation['arrival_rate'] = float(arrival_rate_field.value)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    mouse_pos = pygame.mouse.get_pos()
                    
                    # Handle parameter input field clicks
                    epsilon_field.handle_click(mouse_pos)
                    discount_field.handle_click(mouse_pos)
                    learning_field.handle_click(mouse_pos)
                    
                    # Handle customer simulation input field clicks (right panel)
                    customer_count_field.handle_click(mouse_pos)
                    arrival_rate_field.handle_click(mouse_pos)
                    
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
                    
                    elif start_sim_button.is_clicked(mouse_pos) and not customer_simulation['running']:
                        start_customer_simulation()
                        start_sim_button.enabled = False
                        stop_sim_button.enabled = True
                    
                    elif stop_sim_button.is_clicked(mouse_pos) and customer_simulation['running']:
                        stop_customer_simulation()
                        start_sim_button.enabled = True
                        stop_sim_button.enabled = False
                    
                    elif toggle_heatmap_button.is_clicked(mouse_pos):
                        customer_simulation['show_heatmap'] = not customer_simulation['show_heatmap']
                        toggle_heatmap_button.text = "Hide Heat Map" if customer_simulation['show_heatmap'] else "Show Heat Map"
                    
                    elif reset_heatmap_button.is_clicked(mouse_pos):
                        reset_heatmap()
                    
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
        start_sim_button.update_hover(mouse_pos)
        stop_sim_button.update_hover(mouse_pos)
        
        # Update hover states for goal buttons
        for (goal_pos, goal_button) in get_goal_buttons():
            goal_button.update_hover(mouse_pos)
        
        # Update customer simulation
        update_customer_simulation()
        
        # Check if simulation completed and re-enable start button
        if not customer_simulation['running'] and not start_sim_button.enabled:
            start_sim_button.enabled = True
            stop_sim_button.enabled = False
        
        # Clear screen
        screen.fill(BLACK)
        
        # Update progress bar
        # Show 100% if training is complete, otherwise show actual progress
        if not training and any(q_table.sum() > 0 for q_table in q_tables.values()):
            progress_bar.update(100)  # Show 100% when training completes
        else:
            progress_bar.update(training_progress)
        
        # Draw control panels
        draw_control_panel(screen)
        draw_right_control_panel(screen)
        
        # Draw grid
        draw_grid(screen)
        
        # Draw heat map
        draw_heatmap(screen)
        
        # Update display
        pygame.display.flip()
        clock.tick(FPS)
    
    # Quit
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()