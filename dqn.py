"""
dqn.py
author: andy
"""
import matplotlib.pyplot as plt #c9 update /etc/matplotlibrc backend : Agg to save file as png

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from trader import * #trader.py
from cartpole2 import *

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.9              # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 2000
TRAINING_ITERS = 502 #This should match TraderEnv.trainMax

#102 huge negative rewards
#1004 2.0 negative rewards

#env = TraderEnv(TRAINING_ITERS)
#env = gym.make('CartPole-v0')
#env = CartPoleEnv2()

# env = env.unwrapped

N_ACTIONS = N_STATES = ENV_A_SHAPE = 0.0

# N_ACTIONS = env.action_space.n
# N_STATES = env.observation_space.shape[0]
# ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape

print("N_ACTIONS, N_STATES, ENV_A_SHAPE: ",N_ACTIONS, N_STATES, ENV_A_SHAPE)
class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)#F.relu
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2),dtype='float32')     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        #print("x: ",x)
        
        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

         # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def clearReplayMemory(self):
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2),dtype='float32')
        self.memory_counter = 0                                         # for storing memory
        
#pairs = [["MA","VFC"],["CF","GPS"],
pairs = [["L","POM"],["A","POM"],["FMC","POM"],["AMZN","PEG"],["AMZN","CSCO"],\
        ["AMZN","PAYX"],["AMZN","CHRW"],["AMZN","EXPD"],["AMZN","POM"],["AMZN","SYY"],["AMZN","CCI"],]

#spairs = [["CF","GPS"],["MA","VFC"]]

for p in pairs:
    env = TraderEnv(p[0],p[1],TRAINING_ITERS).unwrapped
    N_ACTIONS = env.action_space.n
    N_STATES = env.observation_space.shape[0]
    ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape

    dqn = DQN()
    
    print("Running ",TRAINING_ITERS," episodes...")
    cum_r = 0.0
    cum_rs = []
    testing = False
    for i_episode in range(TRAINING_ITERS+1):#
        s = env.reset()
        EPSILON = (i_episode / TRAINING_ITERS) * 2 # 
        if(i_episode == (TRAINING_ITERS-1)): 
            print("----LAST TRAINING EPISODE, EPSILON=1.0----")
            EPSILON = 1.0 # last training episode, full greedy
        if(i_episode == TRAINING_ITERS): 
            print("----TRAINING COMPLETE----")
            # print("----CLEARING REPLAY MEMORY PRIOR TO TESTING---")
            # dqn.clearReplayMemory()
            print("----BEGIN TESTING, EPSILON=1.0----")
            EPSILON = 1.0
            testing = True
            #LR = 0.0
            
        #print("s: ",s)
        ep_r = 0
        while True:
            a = dqn.choose_action(s)
    
            # take action
            s_, r, done, info = env.step(a)
    
            #if testing: print("sars: ",s,a,r,s_)
    
            dqn.store_transition(s, a, r, s_)
            
            ep_r += r
            
            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()
                if done and not testing:
                    print('Ep: ', i_episode,'| EPSILON: ', EPSILON,'| Ep_r: ', np.round(ep_r, 2))
                    cum_rs.append(ep_r)
            if done:
                break
            s = s_
            
    #print(len(test_r),len(test_rcolors))
    #print("test_r: ",test_r)
    #print("test_rcolors: ",test_rcolors)
    #pair = "MA_VFC"
    dataLen = "5yr"
    plt.close()
    plt.plot(cum_rs)#,colors=test_rcolors)
    plt.title("Training Episode Returns")
    plt.xlabel("Number of Training Episodes")
    plt.ylabel("Returns")
    plt.savefig(str(p[0])+"_"+str(p[1])+".training_returns.eps"+str(TRAINING_ITERS)+".png")
    plt.show()
    plt.close()
