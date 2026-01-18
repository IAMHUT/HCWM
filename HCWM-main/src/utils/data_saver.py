import pandas as pd
import json
import pickle
import os
from datetime import datetime
import numpy as np


class DataSaver:
    """数据保存工具类"""

    def __init__(self, base_dir='data'):
        self.base_dir = base_dir
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._ensure_dirs()

    def _ensure_dirs(self):
        """确保目录存在"""
        dirs = [
            f'{self.base_dir}/results',
            f'{self.base_dir}/figures',
            f'{self.base_dir}/logs'
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)

    def _convert_to_serializable(self, obj):
        """将对象转换为 JSON 可序列化格式"""
        if isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def save_results_to_excel(self, results, filename=None):
        """保存实验结果到Excel"""
        if filename is None:
            filename = f'results_{self.timestamp}.xlsx'

        filepath = os.path.join(self.base_dir, 'results', filename)

        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # 保存每个算法的详细数据
            for algo_name, data in results.items():
                df = pd.DataFrame({
                    'Episode': data['episodes'],
                    'Average_Reward': data['rewards'],
                    'Std_Reward': data['stds']
                })
                df.to_excel(writer, sheet_name=algo_name[:31], index=False)

            # 保存汇总数据
            summary_data = []
            for algo_name, data in results.items():
                final_rewards = data['rewards'][-10:] if len(data['rewards']) >= 10 else data['rewards']
                summary_data.append({
                    'Algorithm': algo_name,
                    'Final_Avg_Reward': float(pd.Series(final_rewards).mean()),
                    'Final_Std_Reward': float(pd.Series(final_rewards).std()),
                    'Max_Reward': float(pd.Series(data['rewards']).max()),
                    'Episodes_Trained': len(data['episodes'])
                })

            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)

        print(f"✓ Excel数据已保存到: {filepath}")

        # 同时保存CSV格式
        csv_path = filepath.replace('.xlsx', '.csv')
        summary_df.to_csv(csv_path, index=False)
        print(f"✓ CSV汇总数据已保存到: {csv_path}")

        return filepath

    def save_training_losses(self, wm_episodes, wm_losses, img_episodes, img_losses, filename=None):
        """保存训练损失数据"""
        if filename is None:
            filename = f'training_losses_{self.timestamp}.xlsx'

        filepath = os.path.join(self.base_dir, 'results', filename)

        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # 世界模型损失
            if len(wm_losses) > 0:
                wm_df = pd.DataFrame({
                    'Episode': wm_episodes,
                    'World_Model_Loss': wm_losses
                })
                wm_df.to_excel(writer, sheet_name='World_Model_Loss', index=False)

            # 想象策略损失
            if len(img_losses) > 0:
                img_df = pd.DataFrame({
                    'Episode': img_episodes,
                    'Imagination_Policy_Loss': img_losses
                })
                img_df.to_excel(writer, sheet_name='Imagination_Loss', index=False)

        print(f"✓ 训练损失数据已保存到: {filepath}")
        return filepath

    def save_config(self, config, filename=None):
        """保存配置文件（修复JSON序列化问题）"""
        if filename is None:
            filename = f'config_{self.timestamp}.json'

        filepath = os.path.join(self.base_dir, 'logs', filename)

        # 转换为可序列化格式
        serializable_config = self._convert_to_serializable(config)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_config, f, indent=4, ensure_ascii=False)

        print(f"✓ 配置文件已保存到: {filepath}")
        return filepath

    def save_raw_data(self, data, filename=None):
        """保存原始Python对象数据（使用pickle）"""
        if filename is None:
            filename = f'raw_data_{self.timestamp}.pkl'

        filepath = os.path.join(self.base_dir, 'results', filename)

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        print(f"✓ 原始数据已保存到: {filepath}")
        return filepath

    def load_raw_data(self, filepath):
        """加载原始Python对象数据"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data

    def save_comparison_table(self, results, filename=None):
        """保存性能对比表"""
        if filename is None:
            filename = f'comparison_table_{self.timestamp}.xlsx'

        filepath = os.path.join(self.base_dir, 'results', filename)

        # 计算各项指标
        comparison_data = []

        for algo_name, data in results.items():
            rewards = data['rewards']
            episodes = data['episodes']

            # 收敛轮次（达到450的轮次）
            converged_episode = None
            for i, reward in enumerate(rewards):
                if reward >= 450:
                    converged_episode = episodes[i]
                    break

            # 最终10轮平均性能
            final_rewards = rewards[-10:] if len(rewards) >= 10 else rewards
            final_avg = float(pd.Series(final_rewards).mean())
            final_std = float(pd.Series(final_rewards).std())

            comparison_data.append({
                'Algorithm': algo_name,
                'Final_Avg_Reward': round(final_avg, 2),
                'Final_Std': round(final_std, 2),
                'Max_Reward': round(float(pd.Series(rewards).max()), 2),
                'Convergence_Episode': converged_episode if converged_episode else 'Not Converged',
                'Total_Episodes': len(episodes)
            })

        df = pd.DataFrame(comparison_data)

        # 计算相对Vanilla PPO的提升
        vanilla_perf = df[df['Algorithm'] == 'Vanilla PPO']['Final_Avg_Reward'].values[0]
        df['Improvement_vs_Vanilla(%)'] = df['Final_Avg_Reward'].apply(
            lambda x: round(((x - vanilla_perf) / vanilla_perf) * 100, 2)
        )

        df.to_excel(filepath, index=False)
        print(f"✓ 性能对比表已保存到: {filepath}")

        return filepath
