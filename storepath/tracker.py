import numpy as np

environment_rows=11
environment_cols=11

#Numeric action codes: 0=up, 1=right, 2=down, 3=left
actions = ['up', 'right', 'down', 'left']
q_values = np.zeros((environment_rows, environment_cols, 4))



#The array contains 11 rows and 11 columns and each value is initialized to -100
rewards = np.full((environment_rows, environment_cols), -100)
#Sets the reward for the packaging area to 100 (the goal)
rewards[0, 5] = 100

#Define aisle locations
aisles = {}
aisles[1] = [i for i in range(1,10)]
aisles[2] =[1,7,9]
aisles[3] =[i for i in range(1,8)]
aisles[3].append(9)
aisles[4] = [3,7]
aisles[5] = [i for i in range(11)]
aisles[6] = [5]
aisles[7] = [i for i in range(1,10)]
aisles[8] = [3,7]
aisles[9] = [i for i in range(11)]

#Set the rewards for all aisle locations
for row_index in range(1,10):
    for col_index in aisles[row_index]:
        rewards[row_index, col_index] = -1

#Print rewards matrix
for row in rewards:
    print(row)

#Define a function that determines if the specified location is a terminal state
def is_terminal_state(current_row_index, current_col_index):
    #If the reward for this location is -1, then it is not a terminal state (white square)
    if rewards[current_row_index, current_col_index] == -1.:
        return False
    else:
        return True

#Define a function that will choose a random, non-terminal starting location
def get_starting_location():
    #Get a random row and column index
    current_row_index = np.random.randint(environment_rows)
    current_col_index = np.random.randint(environment_cols)
    #Continue choosing random row and column indexes until a non-terminal state is identified.  Until the chosen state is a white square
    while is_terminal_state(current_row_index, current_col_index):
        current_row_index = np.random.randint(environment_rows)
        current_col_index = np.random.randint(environment_cols)
    return current_row_index, current_col_index

#Define an epsilon greedy algorithm that will choose which action to take next (where to move next)
def get_next_action(current_row_index, current_col_index, epsilon):
    #If a random chosen value between 0 and 1 is less than epsilon then choose the most promising value from the Q-table for this state
    if np.random.random() < epsilon:
        return np.argmax(q_values[current_row_index, current_col_index])
    else:#Choose a random action
        return np.random.randint(4)

#Define a function that will get the next location based on the chosen action
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

#Define a function that will get the shortest path between any location within the warehouse that the robot is allowed to travel and the item packaging location
def get_shortest_path(start_row_index, start_col_index):
    #Return immediately if this is an invalid starting location
    if is_terminal_state(start_row_index, start_col_index):
        return []
    else: #If this is a legal starting location
        current_row_index, current_col_index = start_row_index, start_col_index
        shortest_path = []
        shortest_path.append((current_row_index, current_col_index))
        #Continue moving along the path until we reach the goal (item packaging location)
        while not is_terminal_state(current_row_index, current_col_index):
            #Get the best action to take
            action_index = get_next_action(current_row_index, current_col_index, 1.)
            #Move to the next location on the path, and add the new location to the list
            current_row_index, current_col_index = get_next_location(current_row_index, current_col_index, action_index)
            shortest_path.append((current_row_index, current_col_index))
        return shortest_path

#Define training parameters
epsilon = 0.9 #The percentage of time when we should take the best action instead of a random action
discount_factor = 0.9 #Discount factor for future rewards
learning_rate = 0.9 #The rate at which the AI agent should learn

#Run through 1000 episodes
for episode in range(1000):
    #Get the starting location for this episode
    row_index, col_index = get_starting_location()

    #Continue taking actions until we reach a terminal state  (until we reach the item packaging area or crash into an item storage location)
    while not is_terminal_state(row_index, col_index):
        #Choose which action to take
        action_index = get_next_action(row_index, col_index, epsilon)

        #Perform the chosen action and transition to the next state
        old_row, old_col = row_index, col_index
        row_index, col_index = get_next_location(row_index, col_index, action_index)

        #Receive the reward for moving to the new state and calculate the temporal difference
        reward = rewards[row_index, col_index]
        old_q_value = q_values[old_row, old_col, action_index]
        temporal_difference = reward + (discount_factor *  np.max(q_values[row_index, col_index])) - old_q_value

        #Update the Q-Value for the previous state and action pair
        new_q_value = old_q_value + (learning_rate * temporal_difference)
        q_values[old_row, old_col, action_index] = new_q_value

print("Training complete")
print(q_values)
print(get_shortest_path(9,5))