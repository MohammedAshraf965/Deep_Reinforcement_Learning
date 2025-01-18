import gymnasium as gym
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import random
import torch
from torch import nn
import yaml

from dqn import DQN
from experience_replay import ReplayMemory

from datetime import datetime, timedelta
import argparse
import itertools

import flappy_bird_gymnasium
import os

# For printing date and time
DATE_FORMAT = "%m-%d %H:%M:%S"

# Directory for saving run info
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

# "Agg": used to generate plots as images and save them to a file instead of rendering to screen
matplotlib.use('Agg')

# Sometimes GPU is not necessarily faster than cpu. There is an overhead for moving the data to the gpu (sometimes not worth it)
DEVICE = "cuda"  if torch.cuda.is_available() else 'cpu'

# Deep Q-Learning Agent
class Agent:
    def __init__(self, hyperparameter_set):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]

        self.hyperparameter_set = hyperparameter_set

        self.env_ID = hyperparameters['env_ID']
        self.learning_rate = hyperparameters['learning_rate']
        self.discount_factor = hyperparameters['discount_factor']
        self.network_sync_rate = hyperparameters['network_sync_rate']
        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size = hyperparameters['mini_batch_size']
        self.epsilon_init = hyperparameters['epsilon_init']
        self.epsilon_min = hyperparameters['epsilon_min']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.fc1_nodes = hyperparameters['fc1_nodes']
        self.stop_on_reward = hyperparameters['stop_on_reward']
        self.env_make_params = hyperparameters.get['env_make_params', {}]
        self.enable_double_dqn = hyperparameters.get['enable_double_dqn']

        # Neural Network
        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        # Path to Run info
        self.LOG_FILE   = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')

    def run(self, is_training=True, render=False):

        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, 'a') as file:
                file.write(log_message + '\n')
                
        #env = gym.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=False)
        env = gym.make(self.env_ID, render_mode="human" if render else None, **self.env_make_params)

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        # Keep track of rewards per episode
        rewards_per_episode = []

        policy_DQN = DQN(num_states, num_actions, self.fc1_nodes).to(DEVICE)

        if is_training:

            epsilon = self.epsilon_init

            memory = ReplayMemory(self.replay_memory_size)

            target_DQN = DQN(num_states, num_actions, self.fc1_nodes).to(DEVICE)
            target_DQN.load_state_dict(policy_DQN.state_dict())     # Copy weights from policy network to target network

            self.optimizer = torch.optim.Adam(policy_DQN.parameters(), lr=self.learning_rate)

            epsilon_history = []

            # Track number of steps taken. Used for syncing policy --> target network
            step_count = 0

            # Track best reward
            best_reward = -9999999
        
        else:

            # Load learned policy
            policy_DQN.load_state_dict(torch.load(self.MODEL_FILE))

            # Switch model to evaluation mode
            policy_DQN.eval()

        for episode in itertools.count():

            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=DEVICE)
            
            terminated = False              # True when agent reaches goal or fails
            episode_reward = 0.0            # Accumulated reward per episode


            # Perform actions until episode terminates or reaches max rewards
            # (on some envs, it is possible for the agent to train to a point where it NEVER terminates, so stop on reward is necessary)
            while not terminated and episode_reward < self.stop_on_reward:
                
                # Selection based on epsilon-greedy
                if is_training and random.random() < epsilon:
                    # Select random action
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=DEVICE)
                else:
                    # Select best action
                    with torch.no_grad():                                              # No training, evaluating a state
                        # tensor([1, 2, 3, ....]) --> tensor([[1, 2, 3, ...]])
                        action = policy_DQN(state.unsqueeze(dim=0)).squeeze().argmax() # 1D tensor
                        '''
                            state.unsqueeze(dim=0): Pytorch expects a batch layer, so add batch dimension i.e. tensor([1,2,3]) unsqueeze 
                            policy_DQN returns tensor([[1], [2], [3]]), so squeeze it to tensor([1,2,3])
                            .argmax() finds the index of the largest element
                        '''

                # Execute action:
                new_state, reward, terminated, truncated, info = env.step(action.item())
                
                # Accumulate rewards
                episode_reward += reward

                # Convert new state and reward to tensors on device
                new_state = torch.tensor(new_state, dtype=torch.float, device=DEVICE)
                reward = torch.tensor(reward, dtype=torch.float, device=DEVICE)

                if is_training:
                    
                    # Save experience into memory
                    memory.append((state, action, new_state, reward, terminated))

                    # Increment step coutner
                    step_count += 1

                # Move to the next state
                state = new_state

            # Keep track of the rewards collected per episode
            rewards_per_episode.append(episode_reward)
            
            # Save model when new best reward is obtained
            if is_training:
                if episode_reward > best_reward:
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({(episode_reward-best_reward)/best_reward*100:+.1f}%) at episode {episode}, saving model..."
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')
                
                    torch.save(policy_DQN.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward

                
                # Update graph every x seconds
                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(rewards_per_episode, epsilon_history)
                    last_graph_update_time = current_time

            # If enough experience has been collected
            if len(memory) > self.mini_batch_size:
                # Sample from memory
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_DQN, target_DQN)

                # Decay epsilon
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                epsilon_history.append(epsilon)

                # Copy policy network to target network after a certain number of steps
                if step_count > self.network_sync_rate:
                    target_DQN.load_state_dict(policy_DQN.state_dict())
                    step_count = 0

    def save_graph(self, rewards_per_episode, epsilon_history):

        fig = plt.figure(1)

        # Plot episodes (x-axis) vs average rewards (y-axis)
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])
        plt.subplot(121)    # Plot on a 1 row x 2 col grid at cell 1
        plt.plot(rewards_per_episode)
        #plt.xlabel('Episode')
        plt.ylabel('Mean Rewards')
        plt.title('Rewards per Episode')
        plt.plot(mean_rewards)
        
        # Plot episeodes (x-axis) vs epsilon decay (y-axis)
        plt.subplot(122)    # Plot on a 1 row x 2 col grid at cell 2
        # plt.xlabel('Time Steps')
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        # Save plots
        plt.savefig(self.GRAPH_FILE)
        plt.close(fig)

    def optimize(self, mini_batch, policy_dqn, target_dqn):

        # Transpose the list of experiences and seperate each element
        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        # Stack tensors to create batch tensors
        # tensor([[1,2,3]])
        states = torch.stack(states)

        actions = torch.stack(actions)
        
        new_states = torch.stack(new_states)
        
        rewards = torch.stack(rewards)

        terminations = torch.tensor(terminations).float().to(DEVICE)

        with torch.no_grad():

            if self.enable_double_dqn:
                best_actions_from_policy = policy_dqn(new_states).max(dim=1)

                target_q = rewards + (1-terminations) * self.discount_factor * \
                           target_dqn(new_states).gather(dim=1, index=best_actions_from_policy.unsqueeze(dim=1)).squeeze()

            else:
                # Calculate target Q values (expected returns)
                target_q = rewards + (1 - terminations) * self.discount_factor * target_dqn(new_states).max(dim=1)[0]
                
                '''
                    target_dqn(new_states)  ==> tensor([[1,2,3],[4,5,6]])
                    .max(dim=1)             ==> torch.return_types.max(values=tensor([3, 6]), indices=tensor([3, 0, 0, 1]))
                            [0]             ==> tensor([3,6])
                
                '''

        # Calculate Q values from current policy
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
        '''
            policy_dqn(states)      ==> tensor([[1,2,3],[4,5,6]])
            actions.unsqueeze(dim=1)
            .gather(dim=1, actions.unsqueeze(dim=1))  ==> tensor([[1,2,3],[4,5,6]])
                .squeeze()              ==> tensor([1,2,3,4,5,6])

        '''

        # Compute loss
        loss = self.loss_fn(current_q, target_q)

        # Optimize the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

if __name__ == '__main__':
    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test mode.')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    dq1 = Agent(hyperparameter_set=args.hyperparameters)

    if args.train:
        dq1.run(is_training=True)
    else:
        dq1.run(is_training=False, render=True)