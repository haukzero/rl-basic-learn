import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.nn import functional as F
from dqn import smooth_data
from actor_critic import PolicyNet, ValueNet
from trpo import cal_advantage, train


class PPO:
    def __init__(self, d_state, d_action, actor_lr, critic_lr, lmbda, epoches, eps, gamma, device='cpu'):
        self.actor = PolicyNet(d_state, d_action).to(device)
        self.critic = ValueNet(d_state).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.d_state = d_state
        self.d_action = d_action
        self.lmbda = lmbda
        self.epoches = epoches  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.gamma = gamma
        self.device = device

    def take_action(self, states):
        states = torch.from_numpy(states).float().view(-1, self.d_state).to(self.device)
        probs = self.actor(states)
        actor_dist = torch.distributions.Categorical(probs)
        actions = actor_dist.sample().item()
        return actions

    def update(self, states, actions, rewards, next_states, dones):
        states = torch.from_numpy(states).float().view(-1, self.d_state).to(self.device)
        actions = torch.from_numpy(actions).long().view(-1, 1).to(self.device)
        rewards = torch.from_numpy(rewards).float().view(-1, 1).to(self.device)
        next_states = torch.from_numpy(next_states).float().view(-1, self.d_state).to(self.device)
        dones = torch.from_numpy(dones).to(torch.uint8).view(-1, 1).to(self.device)

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = cal_advantage(td_delta.cpu(), self.gamma, self.lmbda).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        for _ in range(self.epoches):
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage  # 截断

            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            critic_loss = (F.mse_loss(self.critic(states), td_target.detach())).mean()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()


if __name__ == '__main__':
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    con_d_action = 16
    dis_action = True
    d_action = env.action_space.n if dis_action else con_d_action
    d_state = env.observation_space.shape[ 0 ]

    actor_lr = 1e-3
    critic_lr = 1e-2
    gamma = 0.98
    lmbda = 0.95
    kl_constraint = 0.0005
    alpha = 0.5
    gamma = 0.98
    lmbda = 0.95
    epoches = 10
    eps = 0.2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = PPO(d_state, d_action, actor_lr, critic_lr,
                lmbda, epoches, eps, gamma, device)

    smooth_window_size = 8
    hist = train(env, agent, 100, dis_action, True)
    hist = smooth_data(hist, smooth_window_size)

    plt.plot(hist)
    plt.xlabel('Epoch')
    plt.ylabel('Reward')
    plt.show()
