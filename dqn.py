import gym
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from collections import deque
from torch.nn import functional as F


def dis_to_con(dis_act, d_action, env):
    low = env.action_space.low[ 0 ]
    high = env.action_space.high[ 0 ]
    return low + (high - low) * dis_act / (d_action - 1)


def smooth_data(data, window_size):
    smoothed_data = [ ]
    for i in range(len(data)):
        window_start = max(0, i - window_size // 2)
        window_end = min(len(data), i + window_size // 2 + 1)
        window_data = data[ window_start:window_end ]
        window_average = np.mean(window_data)
        smoothed_data.append(window_average)
    return smoothed_data


class RelayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def append(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=1):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        return (torch.cat(states, dim=0),
                torch.LongTensor(actions),
                torch.tensor(rewards, dtype=torch.float32),
                torch.cat(next_states, dim=0),
                torch.tensor(dones, dtype=torch.uint8))

    def __len__(self):
        return len(self.buffer)


class QNet(nn.Module):
    def __init__(self, d_state, d_action):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(d_state, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, d_action)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class VNet(nn.Module):
    def __init__(self, d_state, d_action):
        super(VNet, self).__init__()
        self.fc1 = nn.Linear(d_state, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc_A = nn.Linear(512, d_action)
        self.fc_V = nn.Linear(512, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        A = self.fc_A(x)
        V = self.fc_V(x)
        return V + A - A.mean(dim=-1, keepdim=True)


class DQN:
    def __init__(self, d_state, d_action, lr, gamma,
                 target_update_freq=10, device='cpu',
                 net=QNet):
        self.cnt = 0
        self.d_action = d_action
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.device = device
        self.q_net = net(d_state, d_action).to(device)
        self.target_q_net = net(d_state, d_action).to(device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

    def take_action(self, state, t):
        if random.random() < 1 / (t + 1):
            return random.randint(0, self.d_action - 1)
        return self.q_net(state.to(self.device)).argmax().item()

    def _pre_process(self, states, actions, rewards, next_states, dones):
        self.optimizer.zero_grad()
        states = states.to(self.device)
        actions = actions.to(self.device).reshape(-1, 1)
        rewards = rewards.to(self.device).reshape(-1, 1)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device).reshape(-1, 1)
        return states, actions, rewards, next_states, dones

    def _post_process(self, pred, target, loss_func=F.mse_loss):
        loss = loss_func(pred, target).mean()
        loss.backward()
        self.optimizer.step()

        if not (self.cnt % self.target_update_freq):
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.cnt += 1

    def update(self, states, actions, rewards, next_states, dones):
        states, actions, rewards, next_states, dones = self._pre_process(states, actions, rewards, next_states, dones)
        q_vals = self.q_net(states).gather(dim=-1, index=actions)
        max_next_q_vals = self.target_q_net(next_states).max(dim=-1)[ 0 ].reshape(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_vals * (1 - dones)
        self._post_process(q_vals, q_targets)


class DoubleDQN(DQN):
    def __init__(self, d_state, d_action, lr, gamma, target_update_freq=10, device='cpu'):
        super().__init__(d_state, d_action, lr, gamma, target_update_freq, device)

    def update(self, states, actions, rewards, next_states, dones):
        states, actions, rewards, next_states, dones = self._pre_process(states, actions, rewards, next_states, dones)
        q_table = self.q_net(states)
        pred = q_table.gather(dim=-1, index=actions)
        best_a = q_table.argmax(dim=-1, keepdim=True)
        target = (rewards +
                  self.gamma * self.target_q_net(next_states).gather(dim=-1, index=best_a) * (1 - dones))
        self._post_process(pred, target)


def train(env, agent, buffer, minimal_size, batch_size,
          epoches=500, dis_action=True,
          debug=True):
    reward_history = [ ]
    for epoch in range(epoches):
        t = 0
        reward_sum = 0
        state = torch.tensor(env.reset()[ 0 ]).reshape(1, -1)
        done = False
        while not done:
            action = agent.take_action(state, t)
            if dis_action:
                next_state, reward, done, truncated, _ = env.step(action)
            else:
                con_action = dis_to_con(action, agent.d_action, env)
                next_state, reward, done, truncated, _ = env.step(np.array([ con_action ]))
            next_state = torch.tensor(next_state).reshape(1, -1)
            done = done or truncated
            buffer.append(state, action, reward, next_state, done)

            t += 1
            state = next_state
            reward_sum += reward

            if len(buffer) >= minimal_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                agent.update(states, actions, rewards, next_states, dones)

        reward_history.append(reward_sum)
        if debug and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epoches}\tReward {reward_sum}")
    return reward_history


if __name__ == '__main__':
    env = gym.make('Pendulum-v1', render_mode='rgb_array')
    con_d_action = 128
    dis_action = False
    d_action = env.action_space.n if dis_action else con_d_action
    capacity = 10000
    minimal_size = 256
    batch_size = 256
    lr = 1e-3
    gamma = 0.98
    target_update_freq = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    smooth_window_size = 5

    buffer = RelayBuffer(capacity)
    dqn = DQN(env.observation_space.shape[ 0 ],
              d_action,
              lr,
              gamma,
              target_update_freq,
              device)
    double_dqn = DoubleDQN(env.observation_space.shape[ 0 ],
                           d_action,
                           lr,
                           gamma,
                           target_update_freq,
                           device)
    dueling_dqn = DQN(env.observation_space.shape[ 0 ],
                      d_action,
                      lr,
                      gamma,
                      target_update_freq,
                      device,
                      VNet)

    print('==== DQN ====')
    dqn_history = train(env, dqn, buffer,
                        minimal_size, batch_size,
                        epoches=100, dis_action=dis_action, debug=True)
    print('==== Double DQN ====')
    double_dqn_history = train(env, double_dqn, buffer,
                               minimal_size, batch_size,
                               epoches=100, dis_action=dis_action, debug=True)
    print('==== Dueling DQN ====')
    dueling_dqn_history = train(env, double_dqn, buffer,
                                minimal_size, batch_size,
                                epoches=100, dis_action=dis_action, debug=True)

    smooth_dqn = smooth_data(dqn_history, smooth_window_size)
    smooth_double_dqn = smooth_data(double_dqn_history, smooth_window_size)
    smooth_dueling_dqn = smooth_data(dueling_dqn_history, smooth_window_size)

    plt.plot(smooth_dqn, label='DQN')
    plt.plot(smooth_double_dqn, label='Double DQN')
    plt.plot(smooth_dueling_dqn, label='Dueling DQN')
    plt.xlabel('Epoch')
    plt.ylabel('Reward')
    plt.legend()
    plt.show()
