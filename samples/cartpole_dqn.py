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

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size 
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(12, input_dim=self.state_size, activation='relu')) # Input: 4 state values as input
        model.add(Dense(12, activation='relu')) # Hidden layer: two dense layers of 12 perceptrons, each ReLU
        model.add(Dense(self.action_size, activation='linear'))
         # Output layer: estimation of Bellman Q-function for two possible actions (action 1 or action 2) 
         # (linear activation means simply copying the z value as neuron output, since linear just returns the input)
        model.compile(loss='mse',
                      optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
         

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # if there's not enough training --> just takes a random guess as an action, which encourages exploration
        # if its alr trained it finds the action w/ the max Q-function 
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size) 
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])  # returns action
        # once we predict the Q function value of each action --> our final action is just picking the maximum one

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done: # if the episode is finished, done = True, otherwise this part happens
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state, verbose=0)[0])) 
                # the target is set as the sum of the immediate reward and the discounted maximum future reward
                # i.e. the total expected value for taking this action in this state
            target_f = self.model.predict(state, verbose=0) # Gets the current Q-value estimates for the state from the model.
            target_f[0][action] = target # Updates the Q-value for the action taken in the state to the computed target.
            self.model.fit(state, target_f, epochs=1, verbose=0) # training 
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        # If self.epsilon is greater than the minimum epsilon value, it decays by multiplying with self.epsilon_decay. 
        # This gradual decrease in exploration probability helps the agent shift from exploring to exploiting learned strategies as it is replayed or retrained more and more.

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    # DQN Agent only needs to know the dimensions of the state and actions
    # --> The game play will be learned via observing rewards
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0] # gets # of variables describing environment (e.g., position and velocity of the cart and pole).
    action_size = env.action_space.n # gets the number of possible actions the agent can take (e.g., moving left or right).
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = 32 # sets the number of experiences the agent will sample from memory B4 updating the model during each training step

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            # Interaction btw environment and agent
            # env.render()
            action = agent.act(state) # Agent takes action based on state
            next_state, reward, done, _ = env.step(action) # Environement updates state and reward
            env.render()
            reward = reward if not done else -10 # If the episode ends (done is True), set the reward to -10 (penalizing the agent for failing). Otherwise, keep the original reward
            next_state = np.reshape(next_state, [1, state_size]) # reshapes the new state to match the input shape expected by the neural network
            agent.remember(state, action, reward, next_state, done) # Store current experience in memory
            state = next_state 
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        # if e % 10 == 0:
        #     agent.save("./save/cartpole-dqn.h5")
