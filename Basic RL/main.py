import os
import time

import gymnasium

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

### Recipe:
# 1) Load the environment, and test its capabilities using some random decisions.
# 2) Create a model. Use some metrics to determine the quality of the result. Store it
# 3) Retrieve the model and use it to solve the problem in a real case scenario.

### 1) Load the environment, and test its capabilities using some random decisions.
# Gather the enviroment from gymnasium
env_name = 'CartPole-v0'
env = gymnasium.make(env_name, render_mode="human")
# this test the environment using some casual values
episodes = 5
print("Testing the system on {} runs using casual values".format(episodes))
for i in range(episodes):
    state = env.reset()
    terminated = False
    truncated = False
    score = 0
    steps = 0
    while not (terminated or truncated):
        env.render()
        # execute a step using a random value from the action_space.
        # it returns the new state, the reward, the termination status and how it happened (timelimit or bound termination met) and some info
        n_state, reward, terminated, truncated, info= env.step(env.action_space.sample())
        score += reward
        steps += 1
    print("\tEpisode {} scored {} (in {} steps)".format(i + 1, score, steps))

### 2) Create a model. Use some metrics to determine the quality of the result. Store it
# this function renders the environment feasible to the PPO (which requires a vectorized environment to work)
# env = SubprocVecEnv([lambda: make_env() for _ in range(4)]) -> we could use this to work on 4 different env at a time (why? idk)
env = DummyVecEnv([lambda: env])
#Create the model as a MultiLayerPerceptron, define where to to store the logs
model = PPO('MlpPolicy', env, verbose=True, tensorboard_log="./logs")

# this callback evaluates every 10000 steps if the model reached a suitable reward.
# If so, it stops the evaluation. For some reason the threshold gets always reached at the first evaluation.
# The model is therefore not trained enough. Who the F knows why wins a candy. (For this reason, the eval callback is timed out in the corner)
eval_callback = EvalCallback(env, callback_on_new_best=StopTrainingOnRewardThreshold(reward_threshold=200, verbose=1), eval_freq=10000, verbose=1)
model.learn(total_timesteps=20000) #, callback=eval_callback)

# This should evaluate 10 episodes and return the average reward. Helpful to track progress
# and eventually decide to change the hyperparams. Is the second step of a theoretical training (validation)
mean, std_dev = evaluate_policy(model, env, render=True)
print("Evaluating the trained model on 10 episodes:")
print("\tmean: {} - std_dev: {}".format(mean, std_dev))

# store the model into some predefined path
model.save("./models/PPO_first")
model = PPO.load("./models/PPO_first", env=env)

# 3) Retrieve the model and use it to solve the problem in a real case scenario.
# recreate the environment from scratch
print("Testing the system on {} runs using the trained model".format(episodes))
for i in range(episodes):
    obs = env.reset()
    terminated = False
    score = 0
    steps = 0
    while not terminated:
        env.render()
        # here the action is now predicted by the model given the current status of the observation, which is looped and updated
        # with all due peace of the IDE that complains really a lot about this re-assignation
        action, _ = model.predict(obs)
        # for some reason the step function on a VecEnv returns the old tuple, where there is no distinction between the way
        # the model ended its run.
        obs, reward, terminated, _ = env.step(action)
        score += reward
        steps += 1
    print("\tEpisode {} scored {} (in {} steps)".format(i + 1, score, steps))
# remember to close the environment
env.close()