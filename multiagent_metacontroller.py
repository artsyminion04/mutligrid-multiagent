from copy import deepcopy
import gym
from itertools import count
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import torch
from torch.distributions.categorical import Categorical
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import wandb
import ppo
import random

"""
- log one episdoe file
- save model 

- line 158 utils.py?
- env actions take array not dicts
- add first state for 0 values

"""

from utils import plot_single_frame, make_video, extract_mode_from_path

class MultiAgent():
    """This is a meta agent that creates and controls several sub agents. If model_others is True,
    Enables sharing of buffer experience data between agents to allow them to learn models of the 
    other agents. """

    def __init__(self, config, env, device, training=True, with_expert=None, debug=False):
        self.config = config
        self.env = env
        self.device = device
        self.training = training
        self.debug = debug
        self.with_expert = with_expert

        self.agents = []
        self.optimizers = []

        self.envs = gym.vector.SyncVectorEnv([ppo.make_env(config.domain, config.seed, config.capture_video, config.minigrid_mode, config.n_agents)])

        for agent_id in range(env.n_agents):
            agent = ppo.Agent(self.envs).to(device)
            self.agents.append(agent)
            self.optimizers.append(optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5))

        print(f"Environment Details:")
        print(f" - Action space: {self.envs.single_action_space} with {self.envs.single_action_space.n} discrete actions")
        print(f" - Observation space: {self.envs.single_observation_space} with shape {self.envs.single_observation_space.shape}\n")
        print("envs", self.envs.single_observation_space["image"])

        self.batch_size = int(config.num_envs * config.num_steps)
        self.minibatch_size = int(self.batch_size // config.num_minibatches)
        self.num_iterations = config.total_timesteps // self.batch_size

        self.global_steps = 0
        self.memory = [[], []]  # Buffer for storing experiences
    
    def run_one_episode(self, env, episode, device, next_obs, dones, next_done, obs, values, actions, logprobs, rewards, log=True, train=True, save_model=True, visualize=False):
        
        first_obs = next_obs
        if visualize:
            viz_data = self.init_visualization_data(env, first_obs)
        
        if self.config.anneal_lr:
            # Calculates the fraction of total updates remaining
            # Update the learning rate with annealing
            frac = 1.0 - (episode - 1.0) / self.num_iterations
            lrnow = frac * self.config.learning_rate
            for optimizer in self.optimizers:
                optimizer.param_groups[0]["lr"] = lrnow

        # For every step
        # Use policy (ppo.py) to get action based on obs from multigrid env
        # Pass action back to multigrid
        # Get rewards back and update policy
        for step in range(0, self.config.num_steps):
            self.global_steps += 1

            dones[step] = next_done
            for agent_id, agent in enumerate(self.agents):
                #{key: value[agent_id] for key, value in next_obs.items()}
                obs[agent_id, step] = next_obs #s[agent_id]
                with torch.no_grad():
                    action, logprob, _, value= agent.get_action_and_value(next_obs)
                    values[agent_id, step] = value.flatten()
                actions[agent_id, step] = action
                logprobs[agent_id, step] = logprob

            # take action for all agents at the same time
            # dictionary, list, bool
            next_obs, reward, next_done, _ = env.step(action.cpu().numpy())
            if not isinstance(reward, list):
                reward = [reward]

            for agent_id, agent in enumerate(self.agents):
                rewards[agent_id, step] = torch.tensor(reward[agent_id]).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs["image"]).to(device), torch.tensor(next_done, dtype=torch.bool).int().to(device)

        # states is 256 [x,x,x,...]
        # each x is an array [agent1 [15], agent2, agent3...]
        # states = np.array(states)

        # tune policy on action
        self.update_models(self.config, next_obs, next_done, obs, actions, logprobs, values, rewards, dones)

        if visualize:
            viz_data = self.add_visualization_data(viz_data, env, first_obs, actions, next_obs)

        # Logging and checkpointing
        if not log: self.log_one_episode(episode, rewards) #(episode, t, rewards) # change to log
        self.print_terminal_output(episode, torch.sum(rewards))
        self.save_model_checkpoints(episode) 

        if visualize:
            viz_data['rewards'] = np.array(rewards)
            return viz_data

    # def log_one_episode(self, episode, rewards): 
        # print(f"episode={episode}, rewards={rewards}")
        # writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
        # writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

    # agent, full next_obs, rewards/values for just agent, dones
    def bootstrap(self, config, agent, agent_id, next_obs, next_done, rewards, values, dones):
        # bootstrap value if not done
        with torch.no_grad():
            if config.n_agents == 1:
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(self.device)
            else:
                next_value = agent.get_value(next_obs[agent_id]).reshape(1, -1)
                advantages = torch.zeros_like(torch.stack(rewards)).to(self.device)
            lastgaelam = 0
            for t in reversed(range(config.num_steps)):
                if t == config.num_steps - 1:
                    nextnonterminal = 1.0 - next_done.int()
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1].int()
                    nextvalues = values[t + 1]
                delta = rewards[t] + config.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + config.gamma * config.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
        return advantages, returns
   
    def save_model_checkpoints(self, episode):
        if episode % self.config.save_model_episode == 0:
            for i in range(self.config.n_agents):
                print("save model")
                #self.agents[i].save_model()

    def print_terminal_output(self, episode, total_reward):
        if episode % self.config.print_every == 0:
            print('Total steps: {} \t Episode: {} \t Total reward: {}'.format(
                self.config.num_steps, episode, total_reward))

    def init_visualization_data(self, env, state):
        viz_data = {
            'agents_partial_images': [],
            'actions': [],
            'full_images': [],
            'predicted_actions': None
            }
        viz_data['full_images'].append(env.render('rgb_array'))

        if self.config.model_others:
            predicted_actions = []
            predicted_actions.append(self.get_action_predictions(state))
            viz_data['predicted_actions'] = predicted_actions

        return viz_data

    def add_visualization_data(self, viz_data, env, state, actions, next_state):
        viz_data['actions'].append(actions)
        viz_data['agents_partial_images'].append(
            [env.get_obs_render(
                self.get_agent_state(state, i)) for i in range(self.config.n_agents)])
        viz_data['full_images'].append(env.render('rgb_array'))
        if self.config.model_others:
            viz_data['predicted_actions'].append(self.get_action_predictions(next_state))
        return viz_data

    def get_agent_state(self, state, agent_id):
        if self.config.n_agents == 1:
            return state
        return state[agent_id]
        
    def update_models(self, config, next_obs, next_done, obs, actions, logprobs, values, rewards, dones):
        # Don't update model until you've taken enough steps in env
        if self.global_steps > config.initial_memory: 
            if config.num_steps % config.update_every == 0: # How often to update model
                for agent_id, agent in enumerate(self.agents):
                    advantages, returns = self.bootstrap(config, agent, agent_id, next_obs, next_done, rewards[agent_id,:], values[agent_id,:], dones)
                    self.update_policy(config, agent, self.optimizers[agent_id], obs[agent_id,:,:,:,:], actions[agent_id, :], values[agent_id,:], logprobs[agent_id,:], advantages, returns)

    def update_policy(self, config, agent, optimizer, obs, actions, values, logprobs, advantages, returns):
        b_states = obs.reshape((-1,) + obs[0].shape)
        b_logprobs = logprobs.reshape(-1)
        if config.n_agents == 1:
            b_actions = actions.reshape(-1)
        else:
            b_actions = actions.reshape((-1,) + self.env.action_space.shape)

        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(self.batch_size)
        clipfracs = []

        cummulative_loss = 0
        for epoch in range(config.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_states[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # KL Divergence calculation
                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > config.clip_coef).float().mean().item()]

                # Advantage Normalization
                mb_advantages = b_advantages[mb_inds]
                if config.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if config.clip_vloss:
                    # Clipped to prevent large update jumps
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -config.clip_coef,
                        config.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - config.ent_coef * entropy_loss + v_loss * config.vf_coef

                cummulative_loss += pg_loss

                # Gradient Descent
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
                optimizer.step()

            if config.target_kl is not None and approx_kl > config.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        return cummulative_loss, explained_var, clipfracs

    def train(self, env, config, device):
        obs = torch.zeros((config.n_agents, config.num_steps) + env.observation_space["image"].shape).to(device)
        actions = torch.zeros((config.n_agents, config.num_steps) + env.action_space.shape).to(device)
        logprobs = torch.zeros((config.n_agents, config.num_steps, 1)).to(device)
        rewards = torch.zeros((config.n_agents, config.num_steps, 1)).to(device)
        dones = torch.zeros((config.num_steps, 1)).to(device)
        values = torch.zeros((config.n_agents, config.num_steps, 1)).to(device)

        next_obs = env.reset()
        next_obs = torch.Tensor(next_obs["image"]).to(device)
        next_done = torch.zeros(1).to(device)
        
        for episode in range(config.n_episodes):
            if episode % config.visualize_every == 0 and not (self.debug and episode == 0):
                viz_data = self.run_one_episode(env, episode, self.device, next_obs, dones, next_done, obs, values, actions, logprobs, 
                rewards, log=True, train=True, save_model=True, visualize=False)
                # self.visualize(env, self.config.mode + '_training_step' + str(episode), 
                #                viz_data=viz_data)
            else:
                self.run_one_episode(env, episode, self.device, next_obs, dones, next_done, obs, values, actions, logprobs, 
                rewards, log=True, train=True, save_model=True, visualize=False)

        env.close()
        return

    def visualize(self, env, mode, video_dir='videos', viz_data=None):
        if not viz_data:
            viz_data = self.run_one_episode(
                env, episode=0, log=False, train=False, save_model=False, visualize=True)
            env.close()

        video_path = os.path.join(*[video_dir, self.config.experiment_name, self.config.model_name])

        # Set up directory.
        if not os.path.exists(video_path):
            os.makedirs(video_path)

        # Get names of actions
        action_dict = {}
        for act in env.Actions:
            action_dict[act.value] = act.name

        # traj_len =  225 #len(viz_data['rewards'])
        # for agent_id in range(self.n_agents):
        #     for t in range(traj_len):
        #         self.visualize_one_frame(t, viz_data, agent_id, action_dict, video_path, self.config.model_name)
        #         print('Frame {}/{}'.format(t, traj_len))

        # make_video(video_path, mode + '_trajectory_video')

    def visualize_one_frame(self, t, viz_data, agent_id, action_dict, video_path, model_name):
        plot_single_frame(t, 
                          viz_data['full_images'][agent_id][t], 
                          [image[t] for image in viz_data['agents_partial_images']], 
                          viz_data['actions'][agent_id][t], 
                          viz_data['rewards'], 
                          action_dict, 
                          video_path, 
                          self.config.model_name,
                          predicted_actions=viz_data['predicted_actions'])
                        #   all_actions=viz_data['actions'])

    def load_models(self, model_path=None):
        for i in range(self.n_agents):
            if model_path is not None:
                self.agents[i].load_model(save_path=model_path + '_agent_' + str(i))
            else:
                # Use agents' default model path
                self.agents[i].load_model()


