import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class A2C:
    """Advantage Actor-Critic算法"""

    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.gamma = gamma

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

        # 前向传播
        logits = self.policy['actor'](states)
        action_probs = F.softmax(logits, dim=-1)
        values = self.policy['critic'](states)

        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        # 计算优势
        advantages = returns - values.squeeze()

        # 计算损失
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()

        loss = actor_loss + 0.5 * critic_loss - 0.005 * entropy

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()

        return loss.item()
