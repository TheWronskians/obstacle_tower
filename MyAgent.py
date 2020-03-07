from AbstractAgent import AbstractAgent
from dqn.agent import DQNAgent
from dqn.replay_buffer import ReplayBuffer
from dqn.wrappers import *

from environments.obstacle_tower.obstacle_tower_env import ObstacleTowerEnv
from matplotlib import pyplot as plt
import os
import gym
from gym import wrappers
import time
import numpy as np
import random
import itertools

from dqn.model import DQN

import torch
import torch.nn.functional as F
from gym import spaces
from torch.optim import Optimizer

HUMAN_ACTIONS = (0, 6, 12, 18, 21, 24, 30, 36)
# HUMAN_ACTIONS = (0, 3, 6, 12, 18, 21, 24, 30)
NUM_ACTIONS = len(HUMAN_ACTIONS)

# class HumanActionEnv(gym.ActionWrapper):
#     """
#     An environment wrapper that limits the action space to
#     looking left/right, jumping, and moving forward.
#     """
#
#     def __init__(self, env):
#         super().__init__(env)
#         self.actions = HUMAN_ACTIONS
#         self.action_space = gym.spaces.Discrete(len(self.actions))
#
#     def action(self, act):
#         return self.actions[act]

class MyAgent(AbstractAgent):
    def __init__(self, observation_space, action_space):

        # TODO Initialise your agent's models

        shape = observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(shape[-1], shape[0], shape[1]), dtype=np.uint8)
        self.actions = HUMAN_ACTIONS
        self.action_space = gym.spaces.Discrete(len(self.actions))

        self.device = 'cpu'

        self.policy_network = DQN(self.observation_space, self.action_space).to(self.device)

        # for example, if your agent had a Pytorch model it must be load here
        # model.load_state_dict(torch.load( 'path_to_network_model_file', map_location=torch.device(device)))
        model_num = 50
        # self.agent.load_models(model_num)
        self.policy_network.load_state_dict(torch.load('./Models/' + str(model_num) + '_policy.pt',map_location=torch.device(self.device)))

        # self.agent = DQNAgent(
        #     env.observation_space,
        #     env.action_space,
        #     replay_buffer,
        #     use_double_dqn=hyper_params["use-double-dqn"],
        #     lr=hyper_params["learning-rate"],
        #     batch_size=hyper_params["batch-size"],
        #     gamma=hyper_params["discount-factor"]
        # )


    def act(self, state: np.ndarray):
        # Perform processing to observation

        # TODO: return selected action
        # return self.action_space.sample()
        state = np.rollaxis(state,2)
        state = np.array(state) / 255.0
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_network(state)
            _, action = q_values.max(1)
            act = action.item()
            # print('Action: {:}'.format(self.actions[act]))
            return self.actions[act]

if __name__ == "__main__":
    config = {'starting-floor': 0, 'total-floors': 9, 'dense-reward': 1,
              'lighting-type': 0, 'visual-theme': 0, 'default-theme': 0, 'agent-perspective': 1, 'allowed-rooms': 0,
              'allowed-modules': 0,
              'allowed-floors': 0,
              }
    env = ObstacleTowerEnv('./ObstacleTower/obstacletower',
            worker_id=1, retro=True, realtime_mode=True, config=config)

    # env = WarpFrame(env)
    # env = PyTorchFrame(env)
    # env = ClipRewardEnv(env)
    # env = FrameStack(env, 4)

    agent = MyAgent(env.observation_space,env.action_space)

    state = env.reset()
    for t in itertools.count():
        env.render()  # Animate
        action = agent.act(np.array(state))
        next_state, reward, done, _ = env.step(action)
        state = next_state
        if done:
            print('Solved in {} steps'.format(t))
            break
