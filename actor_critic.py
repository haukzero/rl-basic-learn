import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.nn import functional as F
from dqn import smooth_data


class PolicyNet(nn.Module):
    def __init__(self, d_state, d_action):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(d_state, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, d_action)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.softmax(self.fc3(x))


class ValueNet(nn.Module):
    def __init__(self, d_state):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(d_state, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class ActorCritic:
    def __init__(self, d_state, d_action, actor_lr, critic_lr, gamma, device='cpu'):
        self.d_state = d_state
        self.actor = PolicyNet(d_state, d_action).to(device)
        self.critic = ValueNet(d_state).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.device = device

    def take_action(self, states):
        states = torch.from_numpy(states).float().view(-1, self.d_state).to(self.device)
        probs = self.actor(states)
        actor_dist = torch.distributions.Categorical(probs)
        actions = actor_dist.sample().item()
        return actions

    def update(self, states, actions, rewards, next_states):
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        states = torch.from_numpy(states).float().view(-1, self.d_state).to(self.device)
        actions = torch.from_numpy(actions).long().view(-1, 1).to(self.device)
        rewards = torch.from_numpy(rewards).float().view(-1, 1).to(self.device)
        next_states = torch.from_numpy(next_states).float().view(-1, self.d_state).to(self.device)

        critic_prob = self.critic(states)
        critic_target = rewards + self.gamma * self.critic(next_states)
        critic_loss = F.mse_loss(critic_prob, critic_target).mean()
        critic_loss.backward()
        self.critic_optimizer.step()

        delta = critic_target - critic_prob
        log_probs = torch.log(self.actor(states).gather(1, actions))
        actor_loss = (-log_probs * delta.detach()).mean()
        actor_loss.backward()
        self.actor_optimizer.step()


def train(env, agent, epoches=100, debug=True):
    reward_history = [ ]
    for i in range(epoches):
        reward_sum = 0
        states = [ ]
        actions = [ ]
        rewards = [ ]
        next_states = [ ]
        state = env.reset()[ 0 ]
        done = False
        while not done:
            action = agent.take_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            state = next_state

            reward_sum += reward
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        agent.update(states, actions, rewards, next_states)

        reward_history.append(reward_sum)
        if debug and (i + 1) % 10 == 0:
            print(f"Epoch {i + 1}/{epoches}\tReward {reward_sum}")
    return reward_history


if __name__ == '__main__':
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    d_state = env.observation_space.shape[ 0 ]
    d_action = env.action_space.n

    actor_lr = 1e-3
    critic_lr = 1e-2
    gamma = 0.98
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = ActorCritic(d_state, d_action,
                        actor_lr, critic_lr,
                        gamma, device)

    smooth_window_size = 8
    hist = train(env, agent, 100, True)
    hist = smooth_data(hist, smooth_window_size)

    plt.plot(hist)
    plt.xlabel('Epoch')
    plt.ylabel('Reward')
    plt.show()
