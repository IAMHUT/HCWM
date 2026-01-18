import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from ..models.world_model import ImprovedHybridWorldModel
from ..models.policy import ImprovedPPOPolicy
from ..utils.replay_buffer import PrioritizedRegretReplayBuffer
from ..utils.regret import AdvancedRegretComputation


class EnhancedHybridRegretPPO:
    """增强型混合后悔PPO算法"""

    def __init__(
            self,
            state_dim,
            action_dim,
            lr=2e-4,
            gamma=0.99,
            epsilon=0.2,
            epochs=15,
            batch_size=128,
            world_model_train_freq=5,
            use_imagination=True,
            imagination_horizon=15
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_imagination = use_imagination
        self.imagination_horizon = imagination_horizon
        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        self.world_model_train_freq = world_model_train_freq
        self.update_count = 0

        # 初始化网络
        self.policy = ImprovedPPOPolicy(state_dim, action_dim)
        self.imagine_policy = ImprovedPPOPolicy(
            state_dim, action_dim,
            use_rssm=True,
            rssm_state_dim=288  # latent_dim(32) + hidden_dim(256)
        )
        self.world_model = ImprovedHybridWorldModel(state_dim, action_dim)

        # 初始化优化器
        self.optimizer = optim.AdamW(self.policy.parameters(), lr=lr, weight_decay=1e-5)
        self.imagine_optimizer = optim.AdamW(self.imagine_policy.parameters(), lr=lr, weight_decay=1e-5)
        self.world_model_optimizer = optim.AdamW(self.world_model.parameters(), lr=lr * 1.5, weight_decay=1e-5)

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=500)
        self.imagine_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.imagine_optimizer, T_max=500)

        # 工具类
        self.replay_buffer = PrioritizedRegretReplayBuffer(capacity=30000)
        self.regret_computer = AdvancedRegretComputation()

    def collect_trajectory(self, env, max_steps=1000):
        """收集轨迹"""
        trajectory = []
        state, _ = env.reset()
        total_reward = 0

        for _ in range(max_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, log_prob, value, entropy = self.policy.get_action(state_tensor)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            trajectory.append((
                state, action, reward, next_state, done,
                log_prob.item(), value.item(), entropy.item()
            ))

            state = next_state
            if done:
                break

        return trajectory, total_reward

    def _kl_divergence(self, mean1, logvar1, mean2, logvar2):
        """计算KL散度"""
        kl = 0.5 * (
                logvar2 - logvar1 +
                (torch.exp(logvar1) + (mean1 - mean2).pow(2)) / torch.exp(logvar2) - 1
        )
        return kl

    def train_world_model_rssm(self, trajectory, epochs=3):
        """训练RSSM世界模型"""
        if len(trajectory) < 2:
            return None

        # 准备数据
        obs_seq = torch.FloatTensor([t[0] for t in trajectory])
        action_seq = torch.LongTensor([t[1] for t in trajectory])
        reward_seq = torch.FloatTensor([[t[2]] for t in trajectory])
        next_obs_seq = torch.FloatTensor([t[3] for t in trajectory])

        total_loss = 0.0

        for epoch in range(epochs):
            hidden = self.world_model.init_hidden(1)
            latent = self.world_model.init_latent(1)

            recon_loss = 0.0
            reward_loss = 0.0
            kl_loss = 0.0

            for t in range(len(trajectory) - 1):
                obs_t = obs_seq[t:t + 1]
                action_t = F.one_hot(action_seq[t:t + 1], num_classes=self.action_dim).float()
                next_obs_t = next_obs_seq[t:t + 1]
                reward_t = reward_seq[t:t + 1]

                # 前向传播
                outputs = self.world_model.forward_train(obs_t, action_t, hidden, latent)

                # 计算损失
                recon_loss += F.mse_loss(outputs['reconstructed'], next_obs_t)
                reward_loss += F.mse_loss(outputs['predicted_reward'], reward_t)

                kl = self._kl_divergence(
                    outputs['posterior_mean'], outputs['posterior_logvar'],
                    outputs['prior_mean'], outputs['prior_logvar']
                )
                kl_loss += kl.mean()

                # 更新状态
                hidden = outputs['hidden'].detach()
                latent = outputs['latent'].detach()

            # 平均损失
            seq_len = len(trajectory) - 1
            loss = (recon_loss + reward_loss + 0.1 * kl_loss) / seq_len

            # 反向传播
            self.world_model_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 1.0)
            self.world_model_optimizer.step()

            total_loss += loss.item()

        return total_loss / epochs

    def generate_imagined_trajectory(self, start_state, horizon=15):
        """生成想象轨迹"""
        imagined_trajectory = []

        # 初始化
        start_obs = torch.FloatTensor(start_state).unsqueeze(0)
        hidden = self.world_model.init_hidden(1)
        latent = self.world_model.init_latent(1)

        # 编码初始状态
        encoded = self.world_model.encode(start_obs)
        posterior_input = torch.cat([hidden, encoded], dim=1)
        posterior_params = self.world_model.posterior_net(posterior_input)
        posterior_mean, posterior_logvar = torch.chunk(posterior_params, 2, dim=1)
        latent = self.world_model.sample_latent(posterior_mean, posterior_logvar)

        # 展开想象轨迹
        for step in range(horizon):
            current_state = torch.cat([latent, hidden], dim=1)

            # 使用想象策略选择动作
            with torch.no_grad():
                action_probs, value = self.imagine_policy(current_state)
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            # 在想象空间中执行动作
            action_onehot = F.one_hot(action, num_classes=self.action_dim).float()
            outputs = self.world_model.forward_imagine(action_onehot, hidden, latent)

            # 保存经验
            imagined_trajectory.append({
                'state': current_state.detach(),
                'action': action.item(),
                'action_onehot': action_onehot.detach(),
                'reward': outputs['predicted_reward'].detach(),
                'next_state': outputs['state'].detach(),
                'value': value.detach(),
                'log_prob': log_prob.detach(),
                'discount': outputs['predicted_discount'].detach()
            })

            # 更新状态
            hidden = outputs['hidden'].detach()
            latent = outputs['latent'].detach()

        return imagined_trajectory

    def train_imagine_policy(self, real_trajectory, num_imagination_batches=5):
        """训练想象策略"""
        if not self.use_imagination or len(real_trajectory) < 5:
            return 0.0

        total_loss = 0.0

        for _ in range(num_imagination_batches):
            # 从真实轨迹中随机选择起始状态
            start_idx = np.random.randint(0, len(real_trajectory))
            start_state = real_trajectory[start_idx][0]

            # 生成想象轨迹
            imagined_traj = self.generate_imagined_trajectory(
                start_state,
                horizon=self.imagination_horizon
            )

            if len(imagined_traj) == 0:
                continue

            # 计算回报
            returns = []
            R = 0.0
            for t in reversed(imagined_traj):
                R = t['reward'].item() + self.gamma * t['discount'].item() * R
                returns.insert(0, R)

            returns = torch.FloatTensor(returns).unsqueeze(1)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            # 准备数据
            states = torch.cat([t['state'] for t in imagined_traj], dim=0)
            actions = torch.LongTensor([t['action'] for t in imagined_traj])
            old_log_probs = torch.cat([t['log_prob'] for t in imagined_traj], dim=0)
            old_values = torch.cat([t['value'] for t in imagined_traj], dim=0)

            advantages = returns - old_values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # PPO更新
            for _ in range(3):
                action_probs, values = self.imagine_policy(states)
                dist = torch.distributions.Categorical(action_probs)
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy()

                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages.squeeze()
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages.squeeze()

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, returns)
                entropy_bonus = entropy.mean()

                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus

                self.imagine_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.imagine_policy.parameters(), 0.5)
                self.imagine_optimizer.step()

                total_loss += loss.item()

        return total_loss / max(num_imagination_batches * 3, 1)

    def compute_advantages(self, trajectory):
        """计算优势函数"""
        rewards = [t[2] for t in trajectory]
        values = [t[6] for t in trajectory]

        advantages, returns = [], []
        gae = 0.0
        next_value = 0.0
        lambda_gae = 0.97

        for i in reversed(range(len(trajectory))):
            reward, value, done = rewards[i], values[i], trajectory[i][4]

            if done:
                next_value = 0.0

            delta = reward + self.gamma * next_value - value
            gae = delta + self.gamma * lambda_gae * gae * (1 - done)

            advantages.insert(0, gae)
            returns.insert(0, gae + value)
            next_value = value

        return advantages, returns

    def update(self, trajectory):
        """更新策略"""
        # 计算优势和回报
        advantages, returns = self.compute_advantages(trajectory)

        # 计算后悔值
        regrets = self.regret_computer.compute_regret(
            trajectory, self.world_model, self.policy,
            self.action_dim, self.gamma
        )

        # 添加到回放缓冲区
        for i, exp in enumerate(trajectory):
            state, action, reward, next_state, done, log_prob, value, _ = exp
            self.replay_buffer.add(
                state, action, reward, next_state, done,
                log_prob, value, regrets[i], advantages[i], returns[i]
            )

        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        total_loss = 0.0
        beta = min(1.0, 0.4 + 0.6 * self.update_count / 500)

        # PPO更新
        for _ in range(self.epochs):
            batch, weights, indices = self.replay_buffer.sample(
                min(self.batch_size, len(self.replay_buffer)),
                beta=beta
            )

            if len(batch) == 0:
                continue

            # 准备批次数据
            states = torch.FloatTensor(np.array([b[0] for b in batch]))
            actions = torch.LongTensor([b[1] for b in batch])
            old_log_probs = torch.FloatTensor([b[5] for b in batch])
            batch_advantages = torch.FloatTensor([b[8] for b in batch])
            batch_returns = torch.FloatTensor([b[9] for b in batch])
            batch_regrets = torch.FloatTensor([b[7] for b in batch])
            is_weights = torch.FloatTensor(weights)

            # 标准化优势
            batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)

            # 前向传播
            action_probs, values = self.policy(states)
            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy()

            # 计算比率和损失
            ratio = torch.exp(new_log_probs - old_log_probs)

            # 后悔值加权
            regret_weight = 1.0 + batch_regrets
            weighted_advantages = batch_advantages * regret_weight

            # PPO损失
            surr1 = ratio * weighted_advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * weighted_advantages
            policy_loss = -(torch.min(surr1, surr2) * is_weights).mean()
            value_loss = (F.mse_loss(values.squeeze(), batch_returns, reduction='none') * is_weights).mean()
            entropy_bonus = entropy.mean()

            loss = policy_loss + 0.5 * value_loss - 0.02 * entropy_bonus

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

            # 更新优先级
            with torch.no_grad():
                td_errors = (values.squeeze() - batch_returns).abs().cpu().numpy()
                new_priorities = (td_errors + batch_regrets.numpy() + 0.1) ** 0.6
                self.replay_buffer.update_priorities(indices, new_priorities)

            total_loss += loss.item()

        self.update_count += 1
        self.scheduler.step()
        self.imagine_scheduler.step()

        return total_loss / max(self.epochs, 1)
