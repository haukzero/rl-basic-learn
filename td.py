import numpy as np
import matplotlib.pyplot as plt


class CliffWalkEnv:
    def __init__(self, n_rows=4, n_cols=12):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.x = 0
        self.y = n_rows - 1
        self.change = [ [ 0, -1 ], [ 0, 1 ], [ -1, 0 ], [ 1, 0 ] ]

    @property
    def state(self):
        return self.y * self.n_cols + self.x

    def step(self, action):
        self.x = min(self.n_cols - 1, max(0, self.x + self.change[ action ][ 0 ]))
        self.y = min(self.n_rows - 1, max(0, self.y + self.change[ action ][ 1 ]))
        reward = -1
        done = False
        if self.y == self.n_rows - 1 and self.x:
            done = True
            if self.x != self.n_cols - 1:
                reward = -100
        return self.state, reward, done

    def reset(self):
        self.x = 0
        self.y = self.n_rows - 1
        return self.state


class Agent:
    def __init__(self, alpha, gamma, n_rows=4, n_cols=12, n_action=4):
        self.q_table = np.zeros((n_rows * n_cols, n_action))
        self.alpha = alpha
        self.gamma = gamma

    def take_action(self, state, t):
        return np.random.choice(self.q_table.shape[ 1 ]) \
            if np.random.rand() < 0.5 / (t + 1) \
            else self.q_table[ state ].argmax()

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def train(self, *args, **kwargs):
        raise NotImplementedError


class Sarsa(Agent):
    def __init__(self, alpha, gamma, n_rows=4, n_cols=12, n_action=4):
        super().__init__(alpha, gamma, n_rows, n_cols, n_action)

    def update(self, state, action, reward, new_state, new_action):
        td_err = reward + self.gamma * self.q_table[ new_state, new_action ] - self.q_table[ state, action ]
        self.q_table[ state, action ] += self.alpha * td_err

    def train(self, env, epoches=100, max_iter=1000, debug=True):
        reward_history = [ ]
        for i in range(epoches):
            reward_sum = 0
            t = 0
            state = env.reset()
            action = self.take_action(state, t)
            done = False
            while not done and t < max_iter:
                t += 1
                new_state, reward, done = env.step(action)
                new_action = self.take_action(new_state, t)
                self.update(state, action, reward, new_state, new_action)
                state, action = new_state, new_action
                reward_sum += reward
            reward_history.append(reward_sum)
            if debug and (i + 1) % 10 == 0:
                print(f"Epoch {i + 1}/{epoches}\tReward Sum: {reward_sum}")
        return reward_history


class NStepSarsa(Agent):
    def __init__(self, alpha, gamma, n_steps=5, n_rows=4, n_cols=12, n_action=4):
        super().__init__(alpha, gamma, n_rows, n_cols, n_action)
        self.n_steps = n_steps
        self.states = [ ]
        self.actions = [ ]
        self.rewards = [ ]

    def update(self, state, action, reward, new_state, new_action, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        if len(self.states) == self.n_steps:
            G = self.q_table[ new_state, new_action ]
            for i in range(self.n_steps - 1, 0, -1):
                G = self.gamma * G + self.rewards[ i ]
                if done:
                    s = self.states[ i ]
                    a = self.actions[ i ]
                    self.q_table[ s, a ] += self.alpha * (G - self.q_table[ s, a ])
            s = self.states.pop(0)
            a = self.actions.pop(0)
            self.rewards.pop(0)
            self.q_table[ s, a ] += self.alpha * (G - self.q_table[ s, a ])
        if done:
            self.states = [ ]
            self.actions = [ ]
            self.rewards = [ ]

    def train(self, env, epoches=100, max_iter=1000, debug=True):
        reward_history = [ ]
        for i in range(epoches):
            reward_sum = 0
            t = 0
            state = env.reset()
            action = self.take_action(state, t)
            done = False
            while not done and t < max_iter:
                t += 1
                new_state, reward, done = env.step(action)
                new_action = self.take_action(new_state, t)
                self.update(state, action, reward,
                            new_state, new_action, done)
                state, action = new_state, new_action
                reward_sum += reward
            reward_history.append(reward_sum)
            if debug and (i + 1) % 10 == 0:
                print(f"Epoch {i + 1}/{epoches}\tReward Sum: {reward_sum}")
        return reward_history


class QLearning(Agent):
    def __init__(self, alpha, gamma, n_rows=4, n_cols=12, n_action=4):
        super().__init__(alpha, gamma, n_rows, n_cols, n_action)

    def update(self, state, action, reward, new_state):
        td_err = reward + self.gamma * self.q_table[ new_state ].max() - self.q_table[ state, action ]
        self.q_table[ state, action ] += self.alpha * td_err

    def train(self, env, epoches=100, max_iter=1000, debug=True):
        reward_history = [ ]
        for i in range(epoches):
            reward_sum = 0
            t = 0
            state = env.reset()
            done = False
            while not done and t < max_iter:
                action = self.take_action(state, t)
                next_state, reward, done = env.step(action)
                reward_sum += reward
                self.update(state, action, reward, next_state)
                state = next_state
                t += 1
            reward_history.append(reward_sum)
            if debug and (i + 1) % 10 == 0:
                print(f"Epoch {i + 1}/{epoches}\tReward Sum: {reward_sum}")
        return reward_history


def best_act(agent, state):
    q_max = np.max(agent.q_table[ state ])
    a = [ 0 for _ in range(agent.q_table.shape[ 1 ]) ]
    for i in range(agent.q_table.shape[ 1 ]):
        if agent.q_table[ state, i ] == q_max:
            a[ i ] = 1
    return a


def print_agent(agent, env, action_meaning, disaster, end):
    for i in range(env.n_rows):
        for j in range(env.n_cols):
            if (i * env.n_cols + j) in disaster:
                print('****', end=' ')
            elif (i * env.n_cols + j) in end:
                print('EEEE', end=' ')
            else:
                a = best_act(agent, i * env.n_cols + j)
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[ k ] if a[ k ] > 0 else '-'
                print(pi_str, end=' ')
        print()


if __name__ == '__main__':
    alpha = 1e-1
    gamma = 9e-1
    epoches = 500
    n_steps = 4
    env = CliffWalkEnv()
    sarsa = Sarsa(alpha, gamma)
    n_step_sarsa = NStepSarsa(alpha, gamma, n_steps=n_steps)
    qlearning = QLearning(alpha, gamma)

    one_step_rewards = sarsa.train(env, epoches=epoches, debug=False)
    n_steps_rewards = n_step_sarsa.train(env, epoches=epoches, debug=False)
    q_learn_rewards = qlearning.train(env, epoches=epoches, debug=False)
    plt.plot(one_step_rewards, color='blue', label='1 step sarsa')
    plt.plot(n_steps_rewards, color='red', label=f'{n_steps} step sarsa')
    plt.plot(q_learn_rewards, color='green', label='Q Learning')
    plt.legend()
    plt.show()

    action_meaning = [ '^', 'v', '<', '>' ]
    disaster = [ i for i in range((env.n_rows - 1) * env.n_cols + 1, env.n_rows * env.n_cols - 1) ]
    end = [ env.n_rows * env.n_cols - 1 ]
    print('sarsa')
    print_agent(sarsa, env, action_meaning, disaster, end)
    print('n_step sarsa')
    print_agent(n_step_sarsa, env, action_meaning, disaster, end)
    print('q-learning')
    print_agent(qlearning, env, action_meaning, disaster, end)
