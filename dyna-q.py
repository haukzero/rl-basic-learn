import random
import matplotlib.pyplot as plt
from td import CliffWalkEnv, Agent


class DynaQ(Agent):
    def __init__(self, alpha, gamma, n_planning, n_rows=4, n_cols=12, n_action=4):
        super().__init__(alpha, gamma, n_rows, n_cols, n_action)
        self.n_planning = n_planning
        self.model = { }

    def _q_learning(self, state, action, reward, next_state):
        td_err = reward + self.gamma * self.q_table[ next_state ].max() - self.q_table[ state, action ]
        self.q_table[ state, action ] += self.alpha * td_err

    def update(self, state, action, reward, next_state):
        self._q_learning(state, action, reward, next_state)
        self.model[ (state, action) ] = (reward, next_state)
        for _ in range(self.n_planning):
            (s, a), (r, s_) = random.choice(list(self.model.items()))
            self._q_learning(s, a, r, s_)

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
                self.update(state, action, reward, next_state)
                state = next_state
                reward_sum += reward
                t += 1
            reward_history.append(reward_sum)
            if debug and (i + 1) % 10 == 0:
                print(f"Epoch {i + 1}/{epoches}\tReward Sum: {reward_sum}")
        return reward_history


if __name__ == '__main__':
    alpha = 1e-1
    gamma = 9e-1
    epoches = 500
    plannings = [ 0, 2, 20 ]
    env = CliffWalkEnv()

    plt.figure()

    for n_planning in plannings:
        env.reset()
        agent = DynaQ(alpha, gamma, n_planning)
        plt.plot(agent.train(env, epoches, debug=False), label=f"{n_planning}_planning")

    plt.legend()
    plt.show()
