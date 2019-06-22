import gym
from trader import *
import random

# env = gym.make('CartPole-v0')
env = TraderEnv()


q_table = np.zeros([env.observation_space.n, env.action_space.n])
#q_table = {}

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

# For plotting metrics
all_epochs = []
all_penalties = []

for i in range(1, 100001):
    state = env.reset()


    epochs, penalties, reward, = 0, 0, 0
    done = False
    
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space
        else:
            action = np.argmax(q_table[state,:]) # Exploit learned values

        next_state, reward, done, info = env.step(action) 
        
        old_value = q_table[state, action]
        print("state: ",state)
        next_max = np.max(q_table[next_state])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1
        
    if i % 100 == 0:
        clear_output(wait=True)
        print(f"Episode: {i}")

print("Training finished.\n")

'''
for i_episode in range(20):
    observation = env.reset()
    
    
    for t in range(100):
        #env.render()
        # print("observation(state): ",observation)
        action = random.randint(0,2)#
        
        # env.action_space.sample() #just sampling will take a random action
        # print("env.action_space: ",env.action_space)
        # print("observation_space: ",env.observation_space)
        # print("action(0=pushLeft, 1=pushRight): ",action)
        
        observation, reward, done, info = env.step(action)
        print("observation, reward, done, info: ",observation, reward, done, info)
        #input()
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
'''