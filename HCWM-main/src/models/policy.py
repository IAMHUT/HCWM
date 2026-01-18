import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedPPOPolicy(nn.Module):
    """改进的PPO策略网络"""

    def __init__(self, state_dim, action_dim, hidden_dim=256, use_rssm=False, rssm_state_dim=None):
        super().__init__()
        self.use_rssm = use_rssm
        input_dim = rssm_state_dim if use_rssm and rssm_state_dim else state_dim

        # Actor网络
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, action_dim)
        )

        # Critic网络
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, state):
        """前向传播"""
        logits = self.actor(state)
        action_probs = F.softmax(logits, dim=-1)
        value = self.critic(state)
        return action_probs, value

    def get_action(self, state):
        """获取动作"""
        action_probs, value = self.forward(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action.item(), log_prob, value, entropy
