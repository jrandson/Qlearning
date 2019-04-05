from collections import deque

from QLearning.Q_learning import QLearningAgent
from utils import create_uniform_grid, run
from unityagents import UnityEnvironment
import numpy as np
import matplotlib.pyplot as plt

from DQLearning.dqn_agent import Agent

from utils import discretize


def run_sarsa():

    # please do not modify the line below
    env = UnityEnvironment(file_name="./Banana_Linux/Banana.x86_64")
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=True)[brain_name]  # reset the environment

    low, high = min(env_info.vector_observations[0]), max(env_info.vector_observations[0])
    state_grid = create_uniform_grid(low, high, bins=30)
    state_grid = np.array(state_grid)

    q_agent = QLearningAgent(env, state_grid, brain_name, alpha=0.02, gamma=0.99,
                             epsilon=1.0, epsilon_decay_rate=0.9995, min_epsilon=.01, seed=505)

    scores = run(q_agent, env)


def preprocess_state(state, state_grid):
    """Map a continuous state to its discretized representation."""
    return discretize(state, state_grid)


if __name__=="__main__":

    # please do not modify the line below
    env = UnityEnvironment(file_name="./Banana_Linux/Banana.x86_64")
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=True)[brain_name]  # reset the environment

    low, high = min(env_info.vector_observations[0]), max(env_info.vector_observations[0])
    state_grid = create_uniform_grid(low, high, bins=30)
    state_grid = np.array(state_grid)

    q_agent = QLearningAgent(env, state_grid, brain_name, alpha=0.02, gamma=0.99,
                     epsilon=1.0, epsilon_decay_rate=0.9995, min_epsilon=.01, seed=505)

    scores = run(q_agent, env)


