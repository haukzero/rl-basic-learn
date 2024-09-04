import gym
import torch
import matplotlib.pyplot as plt
from torch import nn, optim
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


class REINFORCE:
    def __init__(self, d_state, d_action, lr, gamma, device='cpu'):
        self.device = device
        self.gamma = gamma
        self.net = PolicyNet(d_state, d_action).to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

    def take_action(self, state):
        state = torch.from_numpy(state).float().reshape(1, -1).to(self.device)
        probs = self.net(state)
        action_dist = torch.distributions.Categorical(probs)
        return action_dist.sample().item()

    def update(self, states, actions, rewards):
        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(rewards))):
            state = torch.from_numpy(states[ i ]).float().reshape(1, -1).to(self.device)
            action = torch.LongTensor([actions[i]]).reshape(-1, 1).to(self.device)
            reward = rewards[ i ]
            log_prob = torch.log(self.net(state).gather(1, action))
            G = self.gamma * G + reward
            loss = -log_prob * G
            loss.backward()
        self.optimizer.step()


def train(env, agent, epoches=100, debug=True):
    reward_history = [ ]
    for i in range(epoches):
        reward_sum = 0
        states = [ ]
        actions = [ ]
        rewards = [ ]
        state = env.reset()[ 0 ]
        done = False
        while not done:
            action = agent.take_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state
            reward_sum += reward
        agent.update(states, actions, rewards)
        reward_history.append(reward_sum)
        if debug and (i + 1) % 100 == 0:
            print(f"Epoch {i + 1}/{epoches}\tReward {reward_sum}")
    return reward_history


if __name__ == '__main__':
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    d_state = env.observation_space.shape[ 0 ]
    d_action = env.action_space.n

    lr = 1e-3
    gamma = 0.98
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = REINFORCE(d_state, d_action, lr, gamma, device)

    smooth_window_size = 5
    hist = train(env, agent, 1000, True)
    hist = smooth_data(hist, smooth_window_size)

    plt.plot(hist)
    plt.xlabel('Epoch')
    plt.ylabel('Reward')
    plt.show()
