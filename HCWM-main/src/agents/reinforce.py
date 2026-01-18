import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class REINFORCE:
    """REINFORCE算法"""

    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.gamma = gamma

        self.policy = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def get_action(self, state):
        """获取动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        logits = self.policy(state_tensor)
        action_probs = F.softmax(logits, dim=-1)

        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob

    def collect_trajectory(self, env, max_steps=500):
        """收集轨迹"""
        trajectory = []
        state, _ = env.reset()
        total_reward = 0

        for _ in range(max_steps):
            action, log_prob = self.get_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            trajectory.append((state, action, reward, log_prob.item()))

            state = next_state
            if done:
                break

        return trajectory, total_reward

    def update(self, trajectory):
        """更新策略"""
        if len(trajectory) == 0:
            return 0.0

        # 计算回报
        rewards = [t[2] for t in trajectory]
        returns = []
        R = 0.0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # 准备数据
        states = torch.FloatTensor(np.array([t[0] for t in trajectory]))
        actions = torch.LongTensor([t[1] for t in trajectory])

        # 计算损失
        logits = self.policy(states)
        action_probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        policy_loss = -(log_probs * returns).mean()
        loss = policy_loss - 0.005 * entropy

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()
