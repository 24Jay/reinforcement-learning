import gymnasium as gym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

class Actor(nn.Module):
    def __init__(self, n_state, n_action, hidden_size = 64):
        super(Actor, self).__init__()
        
        self.fc1 = torch.nn.Linear(n_state, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, n_action)

    def forward(self, state):
        x = self.fc1(state)
        x = self.fc2(F.relu(x))
        x = self.fc3(F.relu(x))
        return F.softmax(x, dim=1)

        
class Critic(nn.Module):
    def __init__(self, n_state, hidden_size=64):
        super(Critic, self).__init__()

        self.fc1 = torch.nn.Linear(n_state, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, 1)

    def forward(self, state):
        x = self.fc1(state)
        x = self.fc2(F.relu(x))
        x = self.fc3(F.relu(x))
        return x


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)

class PPO(nn.Module):

    def __init__(self, n_state, n_action, n_hidden = 64, actor_lr=1e-4, critic_lr=1e-4, lmbda=0.1, epochs=10, eps=0.01, gamma=0.99, device="cpu"):
        super(PPO, self).__init__()

        self.actor = Actor(n_state, n_action, hidden_size=n_hidden)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr = actor_lr)

        self.critic = Critic(n_state, hidden_size=n_hidden)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr = critic_lr)

        self.lmbda = lmbda
        self.gamma = gamma
        self.eps = eps
        self.epochs = epochs
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device) 

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta)

        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)

            l1 = ratio * advantage
            l2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            l3 = - torch.min(l1, l2)

            actor_loss = torch.mean(l3)
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

            self.actor_opt.zero_grad()
            self.critic_opt.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_opt.step()
            self.critic_opt.step()
def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(100):
        # with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(10):
            episode_return = 0
            transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
            state, _ = env.reset()
            done = False
            while not done:
                action = agent.take_action(state)
                next_state, reward, done, truncated, info = env.step(action)
                transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                state = next_state
                episode_return += reward

                done = done or truncated
                print(f"-----{i*10 + i_episode}/{num_episodes}: {state} {truncated}")

            return_list.append(episode_return)
            agent.update(transition_dict)
            if (i_episode+1) % 10 == 0:
                print(f"==={i_episode=}, return: {sum(return_list[-10:])/10}")
                # pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
            # pbar.update(1)
    return return_list

actor_lr = 1e-3
critic_lr = 1e-2
num_episodes = 500
hidden = 128

gamma = 0.98
lmbda = 0.95
epochs = 10
eps = 0.2

device = "cpu"
env_name = "CartPole-v1"

env = gym.make(env_name)
# env.seed(0)
torch.manual_seed(0)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = PPO(n_state=state_dim, n_action=action_dim, n_hidden=hidden,actor_lr=actor_lr, critic_lr=critic_lr)

train_on_policy_agent(env, agent, num_episodes)


def test_agent(agent):
    env = gym.make("CartPole-v1", render_mode="human")

    state, info = env.reset()

    print(f"Starting observation: {state}")

    episode_over = False
    total_reward = 0

    while not episode_over:
        # action = env.action_space.sample()  # Random action for now - real agents will be smarter!
        action = agent.take_action(state)
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        episode_over = terminated or truncated

    print(f"Episode finished! Total reward: {total_reward}")
    # env.close()

test_agent(agent)