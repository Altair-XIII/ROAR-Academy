## This is course material for Introduction to Modern Artificial Intelligence
## Example code: cartpole_dqn.py
## Author: Allen Y. Yang
##
## (c) Copyright 2020-2024. Intelligent Racing Inc. Not permitted for commercial use

# Please make sure to install openAI gym module
# pip install gym==0.17.3
# pip install pyglet==1.5.29

import gym
import time

# PID control parameters
Kp = 1.0  # Proportional gain
Ki = 0.0  # Integral gain
Kd = 0.5  # Derivative gain

# Environment settings
EPISODES = 10  # Number of episodes

done = False
while not done:
    env = gym.make('CartPole-v1')
    batch_size = 32
    
    # env.step(1) goes right
    # env.step(0) goes left
    K_p = 1; K_i = 0; K_d = 0.5;  n=500; speed=1.0; cumulative_error = 0
    
    for e in range(EPISODES):
        state = env.reset()
        i, previous_state = 0, 0

        for _ in range(n):
            env.render()
            current_error = state[2] 
            cumulative_error += current_error
            prev_error = previous_state
            diff = current_error - prev_error
            
            action = K_p * current_error + K_d * diff + K_i*cumulative_error

            control_signal =  K_p * current_error + K_d * diff + K_i*cumulative_error

            # if control_signal > 0: 
            #     action = 1
            # else: 
            #     action = 0
            action = 1 if control_signal > 0 else 0 

            next_state, _, done, _ = env.step(action)
            # env.step(action) returns four values: next state, reward, a boolean indicating if the episode has ended, and some extra info
            # here we just use "_" placeholders to ignore the reward and extra info and unpack the next_state and the done boolean
            env.render()

            previous_state = current_error
            state = next_state

            if done:
                done = False
                break