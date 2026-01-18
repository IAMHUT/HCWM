import numpy as np


class PrioritizedRegretReplayBuffer:
    """优先级经验回放缓冲区"""

    def __init__(self, capacity=30000, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.position = 0

    def add(self, state, action, reward, next_state, done, log_prob, value, regret, advantage, ret):
        """添加经验"""
        exp = (state, action, reward, next_state, done, log_prob, value, regret, advantage, ret)
        priority = (abs(regret) + abs(advantage) + 0.1) ** self.alpha

        if len(self.buffer) < self.capacity:
            self.buffer.append(exp)
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = exp
            self.priorities[self.position] = priority

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        """采样经验"""
        if len(self.buffer) == 0:
            return [], None, None

        priorities = np.array(self.priorities[:len(self.buffer)])
        probs = priorities / priorities.sum()

        indices = np.random.choice(
            len(self.buffer),
            size=min(batch_size, len(self.buffer)),
            p=probs,
            replace=False
        )

        samples = [self.buffer[i] for i in indices]

        # 重要性采样权重
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights = weights / weights.max()

        return samples, weights, indices

    def update_priorities(self, indices, priorities):
        """更新优先级"""
        for idx, p in zip(indices, priorities):
            self.priorities[idx] = p

    def __len__(self):
        return len(self.buffer)
