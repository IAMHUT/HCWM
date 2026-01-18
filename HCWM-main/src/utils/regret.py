import torch
import numpy as np


class AdvancedRegretComputation:
    """高级后悔值计算"""

    @staticmethod
    def compute_regret(trajectory, world_model, policy, action_dim, gamma=0.99):
        """
        计算轨迹中每个状态-动作对的后悔值

        Args:
            trajectory: 轨迹数据
            world_model: 世界模型
            policy: 策略网络
            action_dim: 动作空间维度
            gamma: 折扣因子

        Returns:
            regrets: 后悔值列表
        """
        regrets = []

        for i in range(len(trajectory)):
            state, action, reward, next_state, done, _, value, _ = trajectory[i]
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            with torch.no_grad():
                q_values = []
                uncertainties = []

                # 计算所有动作的Q值
                for a in range(action_dim):
                    action_tensor = torch.LongTensor([a])
                    next_state_pred, reward_pred, uncertainty = world_model(state_tensor, action_tensor)
                    _, next_value = policy(next_state_pred)

                    q_value = reward_pred + gamma * next_value * (1 - done)
                    q_values.append(q_value.item())
                    uncertainties.append(uncertainty.item())

                q_values = np.array(q_values)
                uncertainties = np.array(uncertainties)

                # 调整Q值（考虑不确定性）
                adjusted_q_values = q_values - 0.5 * uncertainties

                # 计算后悔值
                max_q = np.max(adjusted_q_values)
                actual_q = adjusted_q_values[action]
                regret = max(0, max_q - actual_q)

                # 归一化
                regret = regret / (abs(max_q) + 1e-8)

            regrets.append(regret)

        return regrets
