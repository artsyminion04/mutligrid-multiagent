import gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ppo
from envs import gym_multigrid

def strtobool (val):
    """Convert a string representation of truth to true (1) or false (0).

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return 1
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return 0
    else:
        raise ValueError("invalid truth value %r" % (val,))

def plot_single_agent_rewards(rewards, steps, n_agents = 1, frame_id=0):

    # Seaborn palette.
    sns.set()
    color_palette = sns.palettes.color_palette()

    # Hardcoded plot settings
    linewidth = 1.25
    ms_current = 9
    xlabelpad = 9
    ylabelpad = 10

    total_subplots_horizontal = 2 + n_agents

    # Determine grid proportions
    full_obs_proportion = 2.0 / total_subplots_horizontal
    agent_proportion = 1.0 / total_subplots_horizontal

    fig, agents_rewards_axes = plt.subplots(n_agents, 1, figsize=(10, n_agents * 2))

    for i in range(n_agents):
        # Cumulative return graphs across bottom right
        cum_return = np.cumsum(rewards)
        agents_rewards_axes.plot(cum_return, color=color_palette[0], lw=linewidth)
        agents_rewards_axes.set_xlabel('Step', fontsize=10, labelpad=xlabelpad)
        agents_rewards_axes.set_ylabel('Agent' + str(i) + ' return', fontsize=10, labelpad=ylabelpad)
        agents_rewards_axes.set_yscale('log')

    fig_path = os.path.join("./plots", "agents_reward.png")
    plt.savefig(fig_path)
    plt.close()


class SingleAgentWrapper(gym.Wrapper):
    """
    Wrapper to convert a multi-agent environment into a
    single-agent environment.
    """

    def __init__(self, env: gym_multigrid):
        """ """
        super().__init__(env)

        self.observation_space = env.observation_space["image"]
        self.action_space = env.action_space

    def reset(self, *args, **kwargs):
        """
        :meta private:
        """
        result = super().reset(*args, **kwargs)
        return tuple(item for item in result)

    def step(self, action):
        """
        :meta private:
        """
        result = super().step({self.agents[0].name: action})
        return tuple(item for item in result)