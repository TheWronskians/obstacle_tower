from environments.obstacle_tower.obstacle_tower_env import ObstacleTowerEnv
from matplotlib import pyplot as plt
import os
import gym
from gym import wrappers
import time
import numpy as np
import random
import itertools

from dqn.agent import DQNAgent
from dqn.replay_buffer import ReplayBuffer
from dqn.wrappers import *
from dqn.model import DQN

# HUMAN_ACTIONS = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33)
HUMAN_ACTIONS = (6, 12, 18, 21, 24, 30, 36)
NUM_ACTIONS = len(HUMAN_ACTIONS)

class HumanActionEnv(gym.ActionWrapper):
    """
    An environment wrapper that limits the action space to
    looking left/right, jumping, and moving forward.
    """

    def __init__(self, env):
        super().__init__(env)
        self.actions = HUMAN_ACTIONS
        self.action_space = gym.spaces.Discrete(len(self.actions))

    def action(self, act):
        return self.actions[act]

def main():
    config = {'starting-floor': 0, 'total-floors': 9, 'dense-reward': 1,
              'lighting-type': 0, 'visual-theme': 0, 'default-theme': 0, 'agent-perspective': 1, 'allowed-rooms': 0,
              'allowed-modules': 0,
              'allowed-floors': 0,
              }
    env = HumanActionEnv(ObstacleTowerEnv('./ObstacleTower/obstacletower',
                           worker_id=1, retro=True, realtime_mode=True, config=config))
    print(env.observation_space)
    print(env.action_space)

    hyper_params = {
        "seed": 6,  # which seed to use
        "replay-buffer-size": int(5e3),  # replay buffer size
        "learning-rate": 1e-4,  # learning rate for Adam optimizer
        "discount-factor": 0.99,  # discount factor
        "num-steps": int(1e6),  # total number of steps to run the environment for
        "batch-size": 32,  # number of transitions to optimize at the same time
        "learning-starts": 5000,  # number of steps before learning starts
        "learning-freq": 1,  # number of iterations between every optimization step
        "use-double-dqn": True,  # use double deep Q-learning
        "target-update-freq": 1000,  # number of iterations between every target network update
        "eps-start": 1.0,  # e-greedy start threshold
        "eps-end": 0.01,  # e-greedy end threshold
        "eps-fraction": 0.05,  # fraction of num-steps
        "print-freq": 10
    }

    np.random.seed(hyper_params["seed"])
    random.seed(hyper_params["seed"])

    #assert "NoFrameskip" in hyper_params["env"], "Require environment with no frameskip"
    #env = gym.make(hyper_params["env"])
    env.seed(hyper_params["seed"])

    #env = NoopResetEnv(env, noop_max=30)
    #env = MaxAndSkipEnv(env, skip=4)
    #env = EpisodicLifeEnv(env)
    #env = FireResetEnv(env)
    # env = WarpFrame(env)
    env = PyTorchFrame(env)
    # env = ClipRewardEnv(env)
    # env = FrameStack(env, 4)

    replay_buffer = ReplayBuffer(hyper_params["replay-buffer-size"])
    # print(env.action_space)
    # action_space = gym.spaces.Discrete({18,19,20,21,22,23,24,25})
    # print(action_space)
    agent = DQNAgent(
        env.observation_space,
        env.action_space,
        replay_buffer,
        use_double_dqn=hyper_params["use-double-dqn"],
        lr=hyper_params["learning-rate"],
        batch_size=hyper_params["batch-size"],
        gamma=hyper_params["discount-factor"]
    )

    eps_timesteps = hyper_params["eps-fraction"] * float(hyper_params["num-steps"])
    episode_rewards = [0.0]
    ep_nums = 0

    state = env.reset()
    for t in range(hyper_params["num-steps"]):
        fraction = min(1.0, float(t) / eps_timesteps)
        eps_threshold = hyper_params["eps-start"] + fraction * (hyper_params["eps-end"] - hyper_params["eps-start"])
        sample = random.random()
        # print(state.shape)
        # TODO
        #  select random action if sample is less equal than eps_threshold
        # take step in env
        # add state, action, reward, next_state, float(done) to reply memory - cast done to float
        # add reward to episode_reward
        if sample > eps_threshold:
            action = agent.act(np.array(state))
        else:
            action = env.action_space.sample()

        next_state, reward, done, _ = env.step(action)
        agent.memory.add(state, action, reward, next_state, float(done))
        state = next_state

        episode_rewards[-1] += reward
        if done:
            state = env.reset()
            episode_rewards.append(0.0)
            ep_nums += 1
            if ep_nums % 50 == 0:
                agent.save_models(ep_nums)
                plot(episode_rewards,ep_nums)




        if t > hyper_params["learning-starts"] and t % hyper_params["learning-freq"] == 0:
            agent.optimise_td_loss()

        if t > hyper_params["learning-starts"] and t % hyper_params["target-update-freq"] == 0:
            agent.update_target_network()

        num_episodes = len(episode_rewards)

        if done and hyper_params["print-freq"] is not None and len(episode_rewards) % hyper_params[
            "print-freq"] == 0:
            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            print("********************************************************")
            print("steps: {}".format(t))
            print("episodes: {}".format(num_episodes))
            print("mean 100 episode reward: {}".format(mean_100ep_reward))
            print("% time spent exploring: {}".format(int(100 * eps_threshold)))
            print("********************************************************")


        #if done and ep_nums % 10 == 0:
        #    animate(env,agent,"anim/progress_"+str(ep_nums))
        #    state = env.reset()

    animate(env,agent,"anim/final")


    env.close()

def plot(episode_rewards,ep_num):
    x = range(len(episode_rewards))
    plt.plot(x,episode_rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Reward per Episode")
    plt.savefig("figs/rewardPlot_"+str(ep_num)+".png")

def animate(env, agent, save_dir):
    """
    Follows (deterministic) greedy policy
    with respect to the given q-value estimator
    and saves animation using openAI gym's Monitor
    wrapper. Monitor will throw an error if monitor
    files already exist in save_dir so use unique
    save_dir for each call.
    """

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    try:
        env = wrappers.Monitor(
            env, save_dir, video_callable=lambda episode_id: True)
    except gym.error.Error as e:
        print(e)

    # Reset the environment
    state = env.reset()
    for t in itertools.count():
        env.render()  # Animate
        action = agent.act(np.array(state))
        next_state, reward, done, _ = env.step(action)
        state = next_state
        if done:
            print('Solved in {} steps'.format(t))
            break

if __name__ == '__main__':
    main()
