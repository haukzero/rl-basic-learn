import numpy as np
import matplotlib.pyplot as plt


class BernoulliBandit:
    def __init__(self, k):
        self.k = k
        self.probs = np.random.uniform(size=k)
        self.best_idx = np.argmax(self.probs)
        self.best_prob = self.probs[ self.best_idx ]

    def step(self, i):
        # 拉动第 i 个拉杆, 随机数比较, 中奖得奖励 1, 否则得 0
        if np.random.rand() <= self.probs[ i ]:
            return 1
        else:
            return 0


class Solver:
    def __init__(self, bandit: BernoulliBandit):
        self.bandit = bandit
        self.counts = np.zeros_like(bandit.probs)
        self.regret = 0
        self.regret_history = [ ]
        self.action_history = [ ]

    def update_regret(self, i):
        self.regret += self.bandit.best_prob - self.bandit.probs[ i ]
        self.regret_history.append(self.regret)

    def step(self):
        raise NotImplementedError

    def run(self, epoches):
        for _ in range(epoches):
            i = self.step()
            self.counts[ i ] += 1
            self.action_history.append(i)
            self.update_regret(i)


class MyAgent(Solver):
    def __init__(self, bandit):
        super().__init__(bandit)
        self.choose_me = np.zeros_like(self.bandit.probs)

    def step(self):
        if np.random.rand() <= 1 / (1 + len(self.action_history)):
            i = np.random.choice(self.bandit.k)
        else:
            if np.random.random() <= 0.5:
                i = np.random.choice(self.bandit.k)
            else:
                i = self.choose_me.argmax()
        r = self.bandit.step(i)
        if r == 1:
            self.choose_me[ i ] += 0.5
        else:
            self.choose_me[ i ] -= 2
        return i


class EpsilonGreedy(Solver):
    def __init__(self, bandit, eps=1e-2, init_prob=1.):
        super().__init__(bandit)
        self.eps = eps
        self.estimate = np.ones_like(self.bandit.probs) * init_prob

    def step(self):
        if np.random.rand() <= self.eps:
            i = np.random.choice(self.bandit.k)
        else:
            i = self.estimate.argmax()
        r = self.bandit.step(i)
        self.estimate[ i ] += 1 / (self.counts[ i ] + 1) * (r - self.estimate[ i ])
        return i


class DecayingEpsilonGreedy(Solver):
    def __init__(self, bandit, init_prob=1.):
        super().__init__(bandit)
        self.estimate = np.ones_like(self.bandit.probs) * init_prob

    def step(self):
        if np.random.rand() <= 1 / (1 + len(self.action_history)):
            i = np.random.choice(self.bandit.k)
        else:
            i = self.estimate.argmax()
        r = self.bandit.step(i)
        self.estimate[ i ] += 1 / (self.counts[ i ] + 1) * (r - self.estimate[ i ])
        return i


class UCB1(Solver):
    def __init__(self, bandit, init_prob=1., coef=1):
        super().__init__(bandit)
        self.coef = coef
        self.estimate = np.ones_like(self.bandit.probs) * init_prob

    def step(self):
        ucb = self.estimate + self.coef * (np.log(self.counts.sum() + 1) / (2 * (self.counts + 1))) ** 0.5
        i = np.argmax(ucb)
        r = self.bandit.step(i)
        self.estimate[ i ] += 1. / (self.counts[ i ] + 1) * (r - self.estimate[ i ])
        return i


class ThompsonSampling(Solver):
    def __init__(self, bandit):
        super().__init__(bandit)
        self.a = np.ones_like(self.bandit.probs)
        self.b = np.ones_like(self.bandit.probs)

    def step(self):
        samples = np.random.beta(self.a, self.b)
        i = samples.argmax()
        r = self.bandit.step(i)
        if r:
            self.a[ i ] += 1
        else:
            self.b[ i ] += 1
        return i


if __name__ == '__main__':
    k = 10
    epoches = 1000
    bandit = BernoulliBandit(k)

    my_agent = MyAgent(bandit)
    eps_greedy = EpsilonGreedy(bandit)
    decay_eps_greedy = DecayingEpsilonGreedy(bandit)
    ucb1 = UCB1(bandit)
    thomson_sampler = ThompsonSampling(bandit)

    my_agent.run(epoches)
    eps_greedy.run(epoches)
    decay_eps_greedy.run(epoches)
    ucb1.run(epoches)
    thomson_sampler.run(epoches)

    print(
        f"累计懊悔:\nmy_agent: {my_agent.regret}"
        f"\t|| eps_greedy: {eps_greedy.regret}"
        f"\ndecay_eps_greedy: {decay_eps_greedy.regret}"
        f"\t|| ucb1: {ucb1.regret}"
        f"\nthomson_sampler: {thomson_sampler.regret}"
    )

    plt.figure()
    plt.subplot(3, 2, 1)
    plt.plot(my_agent.regret_history)
    plt.title('MyAgent Regret History')
    plt.subplot(3, 2, 2)
    plt.plot(eps_greedy.regret_history)
    plt.title('EpsilonGreedy Regret History')
    plt.subplot(3, 2, 3)
    plt.plot(decay_eps_greedy.regret_history)
    plt.title('Decaying EpsilonGreedy Regret History')
    plt.subplot(3, 2, 4)
    plt.plot(ucb1.regret_history)
    plt.title('UCB1 Regret History')
    plt.subplot(3, 2, 5)
    plt.plot(thomson_sampler.regret_history)
    plt.title('ThomsonSampling Regret History')
    plt.suptitle('Regret History')
    plt.tight_layout()
    plt.show()
