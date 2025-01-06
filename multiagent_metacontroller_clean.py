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
- TODO:
- save model
"""

from utils import plot_single_frame, make_video, extract_mode_from_path

class MultiAgent():
    """This is a meta agent that creates and controls several sub agents. If model_others is True,
    Enables sharing of buffer experience data between agents to allow them to learn models of the 
    other agents. """

    def __init__(self, config, envs, device, training=True, with_expert=None, debug=False):
        self.config = config
        self.envs = envs
        self.device = device
        self.training = training
        self.debug = debug
        self.with_expert = with_expert

        print(f"Environment Details:")
        if config.n_agents == 1:
            print(f" - Action space: {self.envs.single_action_space} with {self.envs.single_action_space.n} discrete actions")
        else:
            print(f" - Action space: {self.envs.single_action_space} with {self.envs.single_action_space.high[0] - self.envs.single_action_space.low[0] + 1} discrete actions")
        print(f" - Observation space: {self.envs.single_observation_space}")
        print(f" - Envs: {self.envs.single_observation_space['image']} with shape {self.envs.single_observation_space['image'].shape}\n")
        
        self.agents = []
        self.optimizers = []
        for agent_id in range(config.n_agents):
            agent = ppo.Agent(self.envs, config.n_agents).to(device)
            self.agents.append(agent)
            self.optimizers.append(optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5))

        print("Agent Model Architecture:")
        print(self.agents[0])
        print("\n")

        self.batch_size = int(config.num_envs * config.num_steps)
        self.minibatch_size = int(self.batch_size // config.num_minibatches)
        self.num_iterations = config.total_timesteps // self.batch_size

        self.global_steps = 0
        self.memory = []  # Buffer for storing experiences
                
    def run_one_episode(self, env, episode, device, next_obs, dones, next_done, obs, values, actions, logprobs, rewards, advantages, returns, log=True, train=True, save_model=True, visualize=False):
        
        if visualize:
                viz_data = self.init_visualization_data(env, next_obs)
        
        # Annealing the rate if instructed to do so.
        for agent_id, agent in enumerate(self.agents):            
            if self.config.anneal_lr:
                # Calculates the fraction of total updates remaining
                # Update the learning rate with annealing
                frac = 1.0 - (episode - 1.0) / self.num_iterations
                lrnow = frac * self.config.learning_rate
                self.optimizers[agent_id].param_groups[0]["lr"] = lrnow

        # For every step
        # Use policy (ppo.py) to get action based on obs from multigrid env
        # Pass action back to multigrid
        # Get rewards back and update policy
        for step in range(0, self.config.num_steps):
            self.global_steps += 1
            current_actions = []

            for agent_id, agent in enumerate(self.agents):

                # next_obs = next_obs[0,agent_id]
                # dones = dones[:,:,agent_id]
                # next_done = next_done[0, agent_id], 
                # obs = obs[:, 0, agent_id], 
                # values = values[:,:,agent_id], 
                # actions = actions[:,:,agent_id], 
                # logprobs = logprobs[:,:,agent_id], 
                # rewards = rewards[:,:,agent_id], 

                obs[step, 0, agent_id] = next_obs[0,agent_id]
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value= agent.get_action_and_value(next_obs[0, agent_id])
                    values[step,:,agent_id] = value.flatten()
                actions[step,:,agent_id] = action
                logprobs[step,:, agent_id] = logprob

                current_actions.append(action.cpu().numpy())

            # TRY NOT TO MODIFY: execute the game and log data.
            res = self.envs.step(current_actions)
            full_next_obs, reward, full_next_done, infos = res[0]["image"], res[1], res[2], res[3]

            #next_done = np.logical_or(terminations, truncations)
            rewards[step, 0] = torch.tensor(reward[0]).to(device).view(-1)
            next_obs, next_done = torch.Tensor(full_next_obs).to(device), torch.Tensor(full_next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={self.global_steps}, episodic_return={info['episode']['r']}")

            if visualize:
                viz_data = self.add_visualization_data(viz_data, env, res, current_actions, next_obs)

        # tune policy on action
        self.update_models(episode, self.config, next_obs, next_done, obs, actions, logprobs, values, rewards, dones, advantages, returns)
        
        # Logging and checkpointing
        if log: self.log_one_episode(rewards) 
        self.print_terminal_output(episode, rewards)
        if save_model: self.save_model_checkpoints(episode) 

        if visualize:
            viz_data['rewards'] = np.array(rewards)
            return viz_data, next_obs

        return next_obs

    def log_one_episode(self, rewards):
        wandb.log({'episode_cummulative_reward' : torch.sum(rewards).item()})
        for agent_id in range(len(self.agents)):
            wandb.log({'episode_agent_' + str(agent_id) : torch.sum(rewards[:,:,agent_id]).item()})
    
    def save_model_checkpoints(self, episode):
        if episode % self.config.save_model_episode == 0:
            for i in range(self.config.n_agents):
                self.agent.save_model() 

    def print_terminal_output(self, episode, rewards):
        if episode % self.config.print_every == 0:
            print('Total steps: {} \t Episode: {} \t Total reward: {}'.format(
                self.config.num_steps, episode, torch.sum(rewards).item()))
            for agent_id in range(self.config.n_agents):
                print('Agent ' + str(agent_id) + ' reward: ', torch.sum(rewards[:, :, agent_id]).item())
                
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
        return state[0]["image"][0, agent_id]
        
    def update_models(self, episode, config, next_obs, next_done, obs, actions, logprobs, values, rewards, dones, advantages, returns):

        for agent_id, agent in enumerate(self.agents):
            advantage, ret = self.bootstrap(config, agent, next_obs[0,agent_id], next_done, rewards[:,:,agent_id], 
            values[:,:,agent_id], dones)
            advantages[:,:,agent_id] = advantage
            returns[:,:,agent_id] = ret
            
        self.memory.append([obs, actions, logprobs, values, advantages, returns])

        if self.global_steps > config.initial_memory and episode % config.update_every == 0: # How often to update model
            obs, actions, logprobs, values, advantages, returns = self.process_memory()
            for agent_id, agent in enumerate(self.agents):
                self.update_policy(config, agent, self.optimizers[agent_id], obs[:, 0, agent_id], actions[:,:,agent_id], 
                logprobs[:,:,agent_id], values[:,:,agent_id], advantages[:,:,agent_id], returns[:,:,agent_id])

            self.memory = []

    def process_memory(self):
        res = []
        for tensor_idx in range(len(self.memory[0])):
            tensors = [self.memory[element_idx][tensor_idx] for element_idx in range(len(self.memory))]
            res.append(torch.cat(tensors, dim=0))
        
        return res[0], res[1], res[2], res[3], res[4], res[5]

    def bootstrap(self, config, agent, next_obs, next_done, rewards, values, dones):
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(self.device)
            lastgaelam = 0
            for t in reversed(range(config.num_steps)):
                if t == config.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + config.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + config.gamma * config.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
        return advantages, returns

    def update_policy(self, config, agent, optimizer, obs, actions, logprobs, values, advantages, returns):

        b_obs = obs 
        b_actions = actions.reshape(-1) 
        b_logprobs = logprobs.reshape(-1)
        b_values = values.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        
        # Optimizing the policy and value network
        b_inds = np.arange(self.batch_size)
        clipfracs = []

        cummulative_loss = 0
        for epoch in range(config.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
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

        obs = torch.zeros((config.num_steps, config.num_envs) + self.envs.single_observation_space["image"].shape).to(device)
        actions = torch.zeros((config.num_steps, config.num_envs) + self.envs.single_action_space.shape).to(device)
        logprobs = torch.zeros((config.num_steps, config.num_envs) + (config.n_agents, )).to(device)
        rewards = torch.zeros((config.num_steps, config.num_envs) + (config.n_agents, )).to(device)
        dones = torch.zeros((config.num_steps, config.num_envs)).to(device)
        values = torch.zeros((config.num_steps, config.num_envs) + (config.n_agents, )).to(device)
        advantages = torch.zeros((config.num_steps, config.num_envs) + (config.n_agents, )).to(device)
        returns = torch.zeros((config.num_steps, config.num_envs) + (config.n_agents, )).to(device)

        # TRY NOT TO MODIFY: start the game
        #start_time = time.time()
        next_obs = self.envs.reset()
        next_obs = torch.Tensor(next_obs["image"]).to(device)
        
        import pdb; pdb.set_trace()
        next_done = torch.zeros((config.num_envs)).to(device)
        num_updates = config.total_timesteps // self.batch_size
        
        for episode in range(1, config.n_episodes + 1):
            if episode % config.visualize_every == 0 and not (self.debug and episode == 0):
                viz_data, next_obs = self.run_one_episode(env, episode, self.device, next_obs, dones, next_done, obs, values, actions, logprobs, 
                rewards, advantages, returns, log=True, train=True, save_model=False, visualize=True)
                if np.sum(viz_data["rewards"]) > 0:
                    print("VISUALIZING")
                    self.visualize(env, self.config.mode + '_training_step' + str(episode), viz_data=viz_data)
            else:
                next_obs = self.run_one_episode(env, episode, self.device, next_obs, dones, next_done, obs, values, actions, logprobs, 
                rewards, advantages, returns, log=True, train=True, save_model=False, visualize=False)

        env.close()
        return

    def visualize(self, env, mode, video_dir='videos', viz_data=None):
        if not viz_data:
            viz_data = self.run_one_episode(
                env, episode=0, log=True, train=False, save_model=False, visualize=True)
            env.close()

        video_path = os.path.join(*[video_dir, self.config.experiment_name, self.config.model_name])

        # Set up directory.
        if not os.path.exists(video_path):
            os.makedirs(video_path)

        # Get names of actions
        action_dict = {}
        for act in env.Actions:
            action_dict[act.value] = act.name

        traj_len = len(viz_data['rewards'])
        for t in range(traj_len):
            self.visualize_one_frame(t, viz_data, action_dict, video_path, self.config.model_name)
            # print('Frame {}/{}'.format(t, traj_len))

        make_video(video_path, mode + '_trajectory_video')

    def visualize_one_frame(self, t, viz_data, action_dict, video_path, model_name):
        plot_single_frame(t, 
                          viz_data['full_images'][t], 
                          viz_data['agents_partial_images'][t], 
                          viz_data['actions'][t], 
                          viz_data['rewards'], 
                          action_dict, 
                          video_path, 
                          self.config.model_name, 
                          predicted_actions=viz_data['predicted_actions'])
                          #all_actions=viz_data['actions'])


    def load_models(self, model_path=None):
        for i in range(self.n_agents):
            if model_path is not None:
                self.agents[i].load_model(save_path=model_path + '_agent_' + str(i))
            else:
                # Use agents' default model path
                self.agents[i].load_model()


