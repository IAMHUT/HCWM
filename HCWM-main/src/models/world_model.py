import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedHybridWorldModel(nn.Module):
    """完整的RSSM世界模型"""

    def __init__(self, state_dim, action_dim, hidden_dim=256, latent_dim=32):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # MLP组件(用于后悔值计算)
        self.transition_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, state_dim)
        )

        self.reward_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.uncertainty_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # RSSM组件
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.gru = nn.GRUCell(latent_dim + action_dim, hidden_dim)

        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2)
        )

        self.posterior_net = nn.Sequential(
            nn.Linear(hidden_dim + 128, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, state_dim)
        )

        self.rssm_reward_predictor = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.discount_predictor = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, state, action):
        """前向传播（用于后悔值计算）"""
        if len(action.shape) == 1:
            action = action.long()
            action_onehot = F.one_hot(action, num_classes=self.action_dim).float()
        else:
            action_onehot = action
        x = torch.cat([state, action_onehot], dim=-1)
        next_state_pred = self.transition_net(x)
        reward_pred = self.reward_net(x)
        uncertainty = self.uncertainty_net(x)
        return next_state_pred, reward_pred, uncertainty

    def encode(self, obs):
        """编码观察"""
        return self.encoder(obs)

    def sample_latent(self, mean, logvar):
        """从潜在分布中采样"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward_train(self, obs, prev_action, prev_hidden, prev_latent=None):
        """训练时的前向传播"""
        batch_size = obs.shape[0]
        encoded = self.encode(obs)

        if prev_latent is not None:
            gru_input = torch.cat([prev_latent, prev_action], dim=1)
        else:
            gru_input = torch.zeros(batch_size, self.latent_dim + self.action_dim).to(obs.device)

        hidden = self.gru(gru_input, prev_hidden)

        # 后验网络
        posterior_input = torch.cat([hidden, encoded], dim=1)
        posterior_params = self.posterior_net(posterior_input)
        posterior_mean, posterior_logvar = torch.chunk(posterior_params, 2, dim=1)
        latent = self.sample_latent(posterior_mean, posterior_logvar)

        # 先验网络
        prior_params = self.prior_net(hidden)
        prior_mean, prior_logvar = torch.chunk(prior_params, 2, dim=1)

        # 状态重构和预测
        state = torch.cat([latent, hidden], dim=1)
        reconstructed = self.decoder(state)
        predicted_reward = self.rssm_reward_predictor(state)
        predicted_discount = self.discount_predictor(state)

        return {
            'reconstructed': reconstructed,
            'predicted_reward': predicted_reward,
            'predicted_discount': predicted_discount,
            'latent': latent,
            'hidden': hidden,
            'posterior_mean': posterior_mean,
            'posterior_logvar': posterior_logvar,
            'prior_mean': prior_mean,
            'prior_logvar': prior_logvar
        }

    def forward_imagine(self, prev_action, prev_hidden, prev_latent):
        """想象轨迹时的前向传播"""
        gru_input = torch.cat([prev_latent, prev_action], dim=1)
        hidden = self.gru(gru_input, prev_hidden)

        # 仅使用先验网络
        prior_params = self.prior_net(hidden)
        prior_mean, prior_logvar = torch.chunk(prior_params, 2, dim=1)
        latent = self.sample_latent(prior_mean, prior_logvar)

        state = torch.cat([latent, hidden], dim=1)
        predicted_reward = self.rssm_reward_predictor(state)
        predicted_discount = self.discount_predictor(state)

        return {
            'latent': latent,
            'hidden': hidden,
            'predicted_reward': predicted_reward,
            'predicted_discount': predicted_discount,
            'prior_mean': prior_mean,
            'prior_logvar': prior_logvar,
            'state': state
        }

    def init_hidden(self, batch_size, device='cpu'):
        """初始化隐藏状态"""
        return torch.zeros(batch_size, self.hidden_dim).to(device)

    def init_latent(self, batch_size, device='cpu'):
        """初始化潜在状态"""
        return torch.zeros(batch_size, self.latent_dim).to(device)
