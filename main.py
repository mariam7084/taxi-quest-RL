import random
import gym
import numpy as np

# successfully ran on 28 dec 2024 :)

env = gym.make('Taxi-v3')

# learning rate
alpha = 0.9
gamma = 0.95 #how important the long terms rewards are
epsilon = 1  #randomness or exploration rate; 1 means all the actions will be random
epsilon_decay = 0.9995
min_epsilon = 0.01
num_episodes = 10000 #epochs
max_steps = 100  #number of steps the taxi takes in an epoch, so it doesn't move around in infinite loop

# 5x5 grid -> 25 positions for taxi * 5 diff sqaures * 4diff hotel locations
# so we have 500 states and 4 actions for each state that the car can take(up, down, left, right)
# for each state and each action we will have a q value 

# intializing q_table with zeroes; the parameters below are for shape fo the array
q_table = np.zeros((env.observation_space.n, env.action_space.n))

def choose_action(state):
    if random.uniform (0,1) < epsilon :
        return env.action_space.sample()
    else:
        return np.argmax(q_table[state, :])
    
for episode in range(num_episodes):
    state, _ = env.reset()
    # state = env.reset()[0]

    done = False

    for step in range(max_steps):
        action = choose_action(state)

        next_state, reward, done, truncated, info = env.step(action)

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state, :])

        q_table[state, action] = (1-alpha) *old_value + alpha * (reward + gamma *next_max)

        state = next_state

        if done or truncated:
            break

    epsilon = max(min_epsilon, epsilon * epsilon_decay)


env = gym.make('Taxi-v3', render_mode= 'human')

for episode in range(10):
    state, _ = env.reset()
    done = False
    print('Episode', episode)

    for step in range(max_steps):
        env.render()
        action = np.argmax(q_table[state, :])
        next_state, reward, done, truncated, info = env.step(action)
        state = next_state

        if done or truncated:
            env.render()
            print('Finished Episode', episode, 'with reward', reward)
            break

env.close()
