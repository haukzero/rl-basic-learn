import gym
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.nn import functional as F
from dqn import smooth_data, dis_to_con
from actor_critic import PolicyNet, ValueNet


def cal_advantage(td_deltas, gamma, lmbda):
    advantages = [ ]
    advantage = 0
    td_deltas = td_deltas.detach().numpy()
    for delta in td_deltas[ ::-1 ]:
        advantage = gamma * lmbda * advantage + delta
        advantages.append(advantage)
    advantages.reverse()
    return torch.from_numpy(np.array(advantages)).float()


class TRPO:
    def __init__(self, d_state, d_action, critic_lr, gamma, lmbda, alpha, kl_constraint, device='cpu'):
        self.actor = PolicyNet(d_state, d_action).to(device)
        self.critic = ValueNet(d_state).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.d_state = d_state
        self.d_action = d_action
        self.gamma = gamma
        self.lmbda = lmbda
        self.alpha = alpha
        self.kl_constraint = kl_constraint
        self.device = device

    def take_action(self, states):
        states = torch.from_numpy(states).float().view(-1, self.d_state).to(self.device)
        probs = self.actor(states)
        actor_dist = torch.distributions.Categorical(probs)
        actions = actor_dist.sample().item()
        return actions

    def hessian_matrix_vector_product(self, states, old_action_dists, vector):
        # 计算黑塞矩阵和一个向量的乘积
        new_action_dists = torch.distributions.Categorical(self.actor(states))
        # 计算平均KL距离
        kl = (torch.distributions.kl.kl_divergence(old_action_dists,
                                                   new_action_dists)).mean()
        kl_grad = torch.autograd.grad(kl,
                                      self.actor.parameters(),
                                      create_graph=True)
        kl_grad_vector = torch.cat([ grad.view(-1) for grad in kl_grad ])
        # KL距离的梯度先和向量进行点积运算
        kl_grad_vector_product = torch.dot(kl_grad_vector, vector)
        grad2 = torch.autograd.grad(kl_grad_vector_product,
                                    self.actor.parameters())
        grad2_vector = torch.cat([ grad.view(-1) for grad in grad2 ])
        return grad2_vector

    def conjugate_gradient(self, grad, states, old_action_dists):
        # 共轭梯度法求解方程
        x = torch.zeros_like(grad)
        r = grad.clone()
        p = grad.clone()
        rdotr = torch.dot(r, r)
        for i in range(10):
            Hp = self.hessian_matrix_vector_product(states, old_action_dists, p)
            alpha = rdotr / torch.dot(p, Hp)
            x += alpha * p
            r -= alpha * Hp
            new_rdotr = torch.dot(r, r)
            if new_rdotr < 1e-10:
                break
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
        return x

    def cal_surrogate_obj(self, states, actions, advantage, old_log_probs, actor):
        # 计算策略目标
        log_probs = torch.log(actor(states).gather(1, actions))
        ratio = torch.exp(log_probs - old_log_probs)
        return torch.mean(ratio * advantage)

    def linear_search(self, states, actions, advantage, old_log_probs,
                      old_action_dists, max_vec):
        old_para = torch.nn.utils.convert_parameters.parameters_to_vector(self.actor.parameters())
        old_obj = self.cal_surrogate_obj(states, actions, advantage,
                                         old_log_probs, self.actor)
        for i in range(15):
            coef = self.alpha ** i
            new_para = old_para + coef * max_vec
            new_actor = copy.deepcopy(self.actor)
            torch.nn.utils.convert_parameters.vector_to_parameters(
                new_para, new_actor.parameters())
            new_action_dists = torch.distributions.Categorical(
                new_actor(states))
            kl_div = (torch.distributions.kl.kl_divergence(old_action_dists,
                                                           new_action_dists)).mean()
            new_obj = self.cal_surrogate_obj(states, actions, advantage,
                                             old_log_probs, new_actor)
            if new_obj > old_obj and kl_div < self.kl_constraint:
                return new_para
        return old_para

    def policy_learn(self, states, actions, old_action_dists, old_log_probs, advantage):
        surrogate_obj = self.cal_surrogate_obj(states, actions, advantage,
                                               old_log_probs, self.actor)
        grads = torch.autograd.grad(surrogate_obj, self.actor.parameters())
        obj_grad = torch.cat([ grad.view(-1) for grad in grads ]).detach()
        # 用共轭梯度法计算 x = H^{-1}g
        descent_direction = self.conjugate_gradient(obj_grad, states,
                                                    old_action_dists)

        Hd = self.hessian_matrix_vector_product(states, old_action_dists,
                                                descent_direction)
        max_coef = torch.sqrt(2 * self.kl_constraint / (descent_direction @ Hd + 1e-8))
        new_para = self.linear_search(states, actions, advantage,
                                      old_log_probs,
                                      old_action_dists,
                                      descent_direction * max_coef)
        # 用线性搜索后的参数更新策略
        torch.nn.utils.convert_parameters.vector_to_parameters(
            new_para, self.actor.parameters())

    def update(self, states, actions, rewards, next_states, dones):
        self.critic_optimizer.zero_grad()

        states = torch.from_numpy(states).float().view(-1, self.d_state).to(self.device)
        actions = torch.from_numpy(actions).long().view(-1, 1).to(self.device)
        rewards = torch.from_numpy(rewards).float().view(-1, 1).to(self.device)
        next_states = torch.from_numpy(next_states).float().view(-1, self.d_state).to(self.device)
        dones = torch.from_numpy(dones).to(torch.uint8).view(-1, 1).to(self.device)

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = cal_advantage(td_delta.cpu(), self.gamma, self.lmbda).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()
        old_action_dists = torch.distributions.Categorical(self.actor(states).detach())

        critic_loss = (F.mse_loss(self.critic(states), td_target.detach())).mean()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.policy_learn(states,
                          actions,
                          old_action_dists,
                          old_log_probs,
                          advantage)


def train(env, agent, epoches=100, dis_action=True, debug=True):
    reward_history = [ ]
    for i in range(epoches):
        reward_sum = 0
        states = [ ]
        actions = [ ]
        rewards = [ ]
        next_states = [ ]
        dones = [ ]
        state = env.reset()[ 0 ]
        done = False
        while not done:
            action = agent.take_action(state)
            if dis_action:
                next_state, reward, done, truncated, _ = env.step(action)
            else:
                con_action = dis_to_con(action, agent.d_action, env)
                next_state, reward, done, truncated, _ = env.step(np.array([ con_action ]))
            done = done or truncated
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            state = next_state

            reward_sum += reward
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones, dtype=np.uint8)
        agent.update(states, actions, rewards, next_states, dones)

        reward_history.append(reward_sum)
        if debug and (i + 1) % 10 == 0:
            print(f"Epoch {i + 1}/{epoches}\tReward {reward_sum}")
    return reward_history


if __name__ == '__main__':
    env = gym.make('Pendulum-v1', render_mode='rgb_array')
    con_d_action = 16
    dis_action = False
    d_action = env.action_space.n if dis_action else con_d_action
    d_state = env.observation_space.shape[ 0 ]

    critic_lr = 1e-2
    gamma = 0.98
    lmbda = 0.95
    kl_constraint = 0.0005
    alpha = 0.5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = TRPO(d_state, d_action,
                 critic_lr, gamma,
                 lmbda, alpha,
                 kl_constraint, device)

    smooth_window_size = 8
    hist = train(env, agent, 100, dis_action, True)
    hist = smooth_data(hist, smooth_window_size)

    plt.plot(hist)
    plt.xlabel('Epoch')
    plt.ylabel('Reward')
    plt.show()
