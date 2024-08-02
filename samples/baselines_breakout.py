## This is course material for Introduction to Modern Artificial Intelligence
## Example code: baselines_breakout.py
## Author: Allen Y. Yang
##
## (c) Copyright 2023-2024. Intelligent Racing Inc. Not permitted for commercial use

import os
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import A2C

# There already exists an environment generator that will make and wrap atari environments correctly.

# Initializes the Atari environment 'Breakout-v4' with 8 parallel processes for faster training.
env = make_atari_env('Breakout-v4', n_envs=8, seed=0)

# Stack 4 frames together to provide context over time, i.e. show motion
env = VecFrameStack(env, n_stack=4)

#  Initializes the Actor-Critic model with a CNN policy suited for interpreting Atari environments 
#  + runs the model on CPU
model = A2C('CnnPolicy', env, verbose=1, device='cpu') 
# Bc its A2C --> this model will output a policy that dictates what actions to take

path = os.path.dirname(os.path.abspath(__file__))
model_file_name = path + '/breakout_a2c'
LOAD_PRETRAINED = False
TRAIN_TIMESTEPS = int(1e5) # increase this value to train Atari better 
if LOAD_PRETRAINED:
    # Load saved model, you can use the model w/o having to retrain it 
    model = A2C.load(model_file_name)
else:
  model.learn(total_timesteps=TRAIN_TIMESTEPS)
  model.save(model_file_name)

# Evaluate the agent
episode_reward = 0
obs = env.reset()
for _ in range(1000):
  action, _ = model.predict(obs) # Chooses an action based on the current observation.
  obs, reward, done, info = env.step(action) # Executes the chosen action and returns the next observation, reward, done flag, and additional info.
  env.render("human") # Renders the environment to visualize the agent's performance
  episode_reward += reward
  if sum(done):
    print("Reward:", episode_reward)
    episode_reward = 0.0
    obs = env.reset()