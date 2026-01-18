import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class VanillaPPO:
    """标准PPO算法"""

    def __init__(self, state_dim, action_dim, lr=5e-4, gamma=0.99, epsilon=0.2, epochs=5):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs

        self.policy = nn.ModuleDict({
            'actor': nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim)
            ),
            'critic': nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )
        })

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def get_action(self, state):
        """获取动作"""
        logits = self.policy['actor'](state)
        action_probs = F.softmax(logits, dim=-1)
        value = self.policy['critic'](state)

        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob, value

    def collect_trajectory(self, env, max_steps=500):
        """收集轨迹"""
        trajectory = []
        state, _ = env.reset()
        total_reward = 0

        for _ in range(max_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, log_prob, value = self.get_action(state_tensor)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            trajectory.append((state, action, reward, next_state, done, log_prob.item(), value.item()))

            state = next_state
            if done:
                break

        return trajectory, total_reward

    def compute_advantages(self, trajectory):
        """计算优势函数"""
        rewards = [t[2] for t in trajectory]
        values = [t[6] for t in trajectory]

        advantages, returns = [], []
        gae = 0.0
        next_value = 0.0

        for i in reversed(range(len(trajectory))):
            reward, value, done = rewards[i], values[i], trajectory[i][4]

            if done:
                next_value = 0.0

            delta = reward + self.gamma * next_value - value
            gae = delta + self.gamma * 0.9 * gae

            advantages.insert(0, gae)
            returns.insert(0, gae + value)
            next_value = value

        return advantages, returns

    def update(self, trajectory):
        """更新策略"""
        advantages, returns = self.compute_advantages(trajectory)

        states = torch.FloatTensor(np.array([t[0] for t in trajectory]))
        actions = torch.LongTensor([t[1] for t in trajectory])
        old_log_probs = torch.FloatTensor([t[5] for t in trajectory])
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0.0

        for _ in range(self.epochs):
            logits = self.policy['actor'](states)
            action_probs = F.softmax(logits, dim=-1)
            values = self.policy['critic'](states)

            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values.squeeze(), returns)

            loss = policy_loss + 0.5 * value_loss - 0.005 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / self.epochs
