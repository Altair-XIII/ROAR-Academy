## This is course material for Introduction to Modern Artificial Intelligence
## Example code: cartpole_dqn.py
## Author: Allen Y. Yang
##
## (c) Copyright 2020-2024. Intelligent Racing Inc. Not permitted for commercial use

# Please make sure to install openAI gym module
# pip install gym==0.17.3
# pip install pyglet==1.5.29

import random
import gym
import os
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam

EPISODES = 100
done = False

while not done:
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0] 
    action_size = env.action_space.n 
    batch_size = 32
    
    # env.step(1) goes right
    # env.step(0) goes left
    K_p = 1; K_i = 0; K_d = 0.5;  n=500; speed=1.0
    
    for e in range(100):
        state = env.reset()
        i, previous_state = 0, 0

        for _ in range(n):
            env.render()
            action = int(np.dot([1, 0, 0.5], [state[2], i:= i + state[2]*0.02, (state[2] - previous_state)/0.02]) > 0)
            next_state, reward, done, _ = env.step(action)
            env.render()
            state = next_state
            if done:
                done = False
                break