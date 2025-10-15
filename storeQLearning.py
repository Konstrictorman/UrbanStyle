import numpy as np

# Environment dimensions (matching store.py)
environment_rows = 7
environment_cols = 9

EPISODES = 5000

# Numeric action codes: 0=up, 1=right, 2=down, 3=left
actions = ['up', 'right', 'down', 'left']
q_values = np.zeros((environment_rows, environment_cols, 4))

# The array contains 7 rows and 9 columns and each value is initialized to -100
rewards = np.full((environment_rows, environment_cols), -100)

# Define aisle locations (matching store.py layout)
aisles = {}
aisles[1] = [1, 2, 3, 4, 5, 6, 7]  # Row 1: columns 1-7 are aisles
aisles[2] = [1, 4, 7]              # Row 2: only columns 1 and 7 are aisles
aisles[3] = [1, 2, 3, 4, 5, 6, 7]  # Row 3: columns 1-7 are aisles
aisles[4] = [1, 4, 7]              # Row 4: only columns 1 and 7 are aisles
aisles[5] = [1, 2, 3, 4, 5, 6, 7]  # Row 5: columns 1-7 are aisles

# Set the rewards for all aisle locations
for row_index in range(environment_rows):
    if row_index in aisles:
        for col_index in aisles[row_index]:
            rewards[row_index, col_index] = -1

# Set entrance at position (1,0) - high reward for entrance
rewards[1, 0] = 100
rewards[5, 8] = 100  # Another entrance

# Set goal locations with maximum rewards
rewards[1, 7] = 99   # Goal 1
rewards[3, 7] = 100  # Goal 2 (highest priority)
rewards[5, 2] = 101  # Goal 3 (highest reward)

# Define a function that determines if the specified location is a terminal state
def is_terminal_state(current_row_index, current_col_index):
    # If the reward for this location is -1, then it is not a terminal state (white square)
    if rewards[current_row_index, current_col_index] == -1.:
        return False
    else:
        return True

# Define a function that will choose a random, non-terminal starting location
def get_starting_location():
    # Get a random row and column index
    current_row_index = np.random.randint(environment_rows)
    current_col_index = np.random.randint(environment_cols)
    # Continue choosing random row and column indexes until a non-terminal state is identified
    while is_terminal_state(current_row_index, current_col_index):
        current_row_index = np.random.randint(environment_rows)
        current_col_index = np.random.randint(environment_cols)
    return current_row_index, current_col_index

# Define an epsilon greedy algorithm that will choose which action to take next
def get_next_action(current_row_index, current_col_index, epsilon):
    # If a random chosen value between 0 and 1 is less than epsilon then choose the most promising value from the Q-table
    if np.random.random() < epsilon:
        return np.argmax(q_values[current_row_index, current_col_index])
    else:  # Choose a random action
        return np.random.randint(4)

# Define a function that will get the next location based on the chosen action
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

# Define a function that will get the shortest path to any goal location
def get_shortest_path_to_any_goal(start_row_index, start_col_index):
    # Return immediately if this is an invalid starting location
    if is_terminal_state(start_row_index, start_col_index):
        return []
    else:  # If this is a legal starting location
        current_row_index, current_col_index = start_row_index, start_col_index
        shortest_path = []
        shortest_path.append((current_row_index, current_col_index))
        visited = set()
        visited.add((current_row_index, current_col_index))
        
        # Continue moving along the path until we reach any goal
        max_steps = 30  # Prevent infinite loops
        step_count = 0
        
        while not is_terminal_state(current_row_index, current_col_index) and step_count < max_steps:
            # Get the best action to take (exploit learned policy)
            action_index = np.argmax(q_values[current_row_index, current_col_index])
            
            # Try to move in the direction of the best action
            new_row, new_col = get_next_location(current_row_index, current_col_index, action_index)
            
            # If the best action leads to a visited location or invalid location, try other actions
            if (new_row, new_col) in visited or rewards[new_row, new_col] == -100:
                # Try all actions and pick the best valid one
                best_action = -1
                best_value = float('-inf')
                for action in range(4):
                    test_row, test_col = get_next_location(current_row_index, current_col_index, action)
                    if (test_row, test_col) not in visited and rewards[test_row, test_col] != -100:
                        if q_values[current_row_index, current_col_index, action] > best_value:
                            best_value = q_values[current_row_index, current_col_index, action]
                            best_action = action
                
                if best_action != -1:
                    new_row, new_col = get_next_location(current_row_index, current_col_index, best_action)
                else:
                    break  # No valid moves available
            
            # Move to the next location on the path
            current_row_index, current_col_index = new_row, new_col
            shortest_path.append((current_row_index, current_col_index))
            visited.add((current_row_index, current_col_index))
            step_count += 1
            
        return shortest_path

# Define a function to find the path to a specific goal
def get_shortest_path_to_goal(start_row_index, start_col_index, goal_row, goal_col):
    # Return immediately if this is an invalid starting location
    if is_terminal_state(start_row_index, start_col_index):
        return []
    else:
        current_row_index, current_col_index = start_row_index, start_col_index
        shortest_path = []
        shortest_path.append((current_row_index, current_col_index))
        visited = set()
        visited.add((current_row_index, current_col_index))
        
        max_steps = 30
        step_count = 0
        
        # Continue until we reach the specific goal
        while (current_row_index != goal_row or current_col_index != goal_col) and step_count < max_steps:
            # Get the best action to take (exploit learned policy)
            action_index = np.argmax(q_values[current_row_index, current_col_index])
            
            # Try to move in the direction of the best action
            new_row, new_col = get_next_location(current_row_index, current_col_index, action_index)
            
            # If the best action leads to a visited location or invalid location, try other actions
            if (new_row, new_col) in visited or rewards[new_row, new_col] == -100:
                # Try all actions and pick the best valid one
                best_action = -1
                best_value = float('-inf')
                for action in range(4):
                    test_row, test_col = get_next_location(current_row_index, current_col_index, action)
                    if (test_row, test_col) not in visited and rewards[test_row, test_col] != -100:
                        if q_values[current_row_index, current_col_index, action] > best_value:
                            best_value = q_values[current_row_index, current_col_index, action]
                            best_action = action
                
                if best_action != -1:
                    new_row, new_col = get_next_location(current_row_index, current_col_index, best_action)
                else:
                    break  # No valid moves available
            
            # Move to the next location on the path
            current_row_index, current_col_index = new_row, new_col
            shortest_path.append((current_row_index, current_col_index))
            visited.add((current_row_index, current_col_index))
            step_count += 1
            
        return shortest_path

# Define training parameters
epsilon = 0.9  # The percentage of time when we should take the best action instead of a random action
discount_factor = 0.9  # Discount factor for future rewards
learning_rate = 0.9  # The rate at which the AI agent should learn

print("Rewards Matrix:")
for row in rewards:
    print(row)
print()

# Run through N episodes (more episodes for multiple goals)
for episode in range(EPISODES):
    # Get the starting location for this episode
    row_index, col_index = get_starting_location()

    # Continue taking actions until we reach a terminal state
    max_steps_per_episode = 50
    step_count = 0
    
    while not is_terminal_state(row_index, col_index) and step_count < max_steps_per_episode:
        # Choose which action to take
        action_index = get_next_action(row_index, col_index, epsilon)

        # Perform the chosen action and transition to the next state
        old_row, old_col = row_index, col_index
        new_row, new_col = get_next_location(row_index, col_index, action_index)
        
        # Only update if the move is valid (not hitting walls)
        if rewards[new_row, new_col] != -100:
            row_index, col_index = new_row, new_col

        # Receive the reward for moving to the new state and calculate the temporal difference
        reward = rewards[row_index, col_index]
        old_q_value = q_values[old_row, old_col, action_index]
        
        # Calculate temporal difference
        if is_terminal_state(row_index, col_index):
            # If we reached a goal, no future reward
            temporal_difference = reward - old_q_value
        else:
            # If we're still in a non-terminal state, consider future rewards
            temporal_difference = reward + (discount_factor * np.max(q_values[row_index, col_index])) - old_q_value

        # Update the Q-Value for the previous state and action pair
        new_q_value = old_q_value + (learning_rate * temporal_difference)
        q_values[old_row, old_col, action_index] = new_q_value
        
        step_count += 1
    
    # Decay epsilon over time for better convergence
    if episode % 500 == 0 and episode > 0:
        epsilon = max(0.1, epsilon * 0.95)

print("Training complete!")
print("\nFinal Q-values for each action (up, right, down, left):")
for row in range(environment_rows):
    for col in range(environment_cols):
        if rewards[row, col] == -1:  # Only show for walkable areas
            print(f"Position ({row},{col}): {q_values[row, col]}")

# Test paths to different goals
print("\n" + "="*50)
print("PATH TESTING")
print("="*50)

# Test paths from various starting positions to different goals
test_starts = [(1, 1), (3, 1), (5, 1)]
goals = [(1, 7), (3, 7), (5, 2)]

for start in test_starts:
    print(f"\nFrom position {start}:")
    
    # Path to any goal
    path_to_any = get_shortest_path_to_any_goal(start[0], start[1])
    if path_to_any:
        goal_reached = path_to_any[-1]
        goal_reward = rewards[goal_reached[0], goal_reached[1]]
        print(f"  Path to any goal (reward {goal_reward}): {path_to_any}")
    
    # Path to specific goals
    for goal in goals:
        path = get_shortest_path_to_goal(start[0], start[1], goal[0], goal[1])
        if path and len(path) > 1:
            print(f"  Path to goal {goal}: {path}")

print("\n" + "="*50)
print("GOAL ANALYSIS")
print("="*50)

# Analyze which goals are most accessible
goal_positions = [(1, 7, 99), (3, 7, 100), (5, 2, 101)]
for goal_row, goal_col, reward_val in goal_positions:
    print(f"\nGoal at ({goal_row}, {goal_col}) with reward {reward_val}:")
    
    # Test accessibility from different starting points
    for start_row in range(1, 6):
        for start_col in [1, 4, 7]:  # Common aisle positions
            if rewards[start_row, start_col] == -1:  # Valid starting position
                path = get_shortest_path_to_goal(start_row, start_col, goal_row, goal_col)
                if path and len(path) > 1:
                    print(f"  Accessible from ({start_row}, {start_col}) in {len(path)-1} steps")
