import sys

sys.path.append('..')

import gymnasium as gym
import torch
import torch.nn.functional as F
import numpy as np

from src.utils.data_saver import DataSaver
from src.agents.hybrid_ppo import EnhancedHybridRegretPPO
from src.agents.vanilla_ppo import VanillaPPO
from src.agents.reinforce import REINFORCE
from src.agents.a2c import A2C
from src.visualization.plotter import plot_selected_metrics
from src.utils.common import set_seed


def test_policy(env, agent, num_episodes=10, agent_type='ppo'):
    """测试策略性能"""
    total_rewards = []

    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            with torch.no_grad():
                if agent_type == 'hybrid':
                    action_probs, _ = agent.policy(state_tensor)
                    action = torch.argmax(action_probs, dim=1).item()
                elif agent_type == 'ppo':
                    logits = agent.policy['actor'](state_tensor)
                    action_probs = F.softmax(logits, dim=-1)
                    action = torch.argmax(action_probs, dim=1).item()
                elif agent_type == 'reinforce':
                    logits = agent.policy(state_tensor)
                    action_probs = F.softmax(logits, dim=-1)
                    action = torch.argmax(action_probs, dim=1).item()
                elif agent_type == 'a2c':
                    logits = agent.policy['actor'](state_tensor)
                    action_probs = F.softmax(logits, dim=-1)
                    action = torch.argmax(action_probs, dim=1).item()

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        total_rewards.append(episode_reward)

    return np.mean(total_rewards), np.std(total_rewards)


def train_and_compare():
    """训练并对比所有算法"""

    # 设置随机种子
    set_seed(42)

    # 创建环境
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # 参数设置
    num_episodes = 500
    test_interval = 10

    print("=" * 80)
    print("初始化算法...")
    print("=" * 80)

    # 初始化所有算法
    hybrid_agent = EnhancedHybridRegretPPO(
        state_dim, action_dim,
        use_imagination=True,
        imagination_horizon=15
    )
    vanilla_ppo = VanillaPPO(state_dim, action_dim)
    reinforce_agent = REINFORCE(state_dim, action_dim)
    a2c_agent = A2C(state_dim, action_dim)

    # 结果存储
    results = {
        'Hybrid Regret PPO': {'episodes': [], 'rewards': [], 'stds': []},
        'Vanilla PPO': {'episodes': [], 'rewards': [], 'stds': []},
        'REINFORCE': {'episodes': [], 'rewards': [], 'stds': []},
        'A2C': {'episodes': [], 'rewards': [], 'stds': []}
    }

    world_model_loss_episodes = []
    world_model_losses = []
    imagination_loss_episodes = []
    imagination_losses = []

    print("\n开始训练对比实验...")
    print("=" * 80)

    # 训练 Hybrid Regret PPO
    print("\n[1/4] 训练 Enhanced Hybrid Regret PPO (with Imagination)...")
    for episode in range(num_episodes):
        trajectory, total_reward = hybrid_agent.collect_trajectory(env)

        # 训练世界模型
        if episode % hybrid_agent.world_model_train_freq == 0:
            wm_loss = hybrid_agent.train_world_model_rssm(trajectory, epochs=3)
            if wm_loss is not None:
                world_model_loss_episodes.append(episode)
                world_model_losses.append(wm_loss)

        # 训练想象策略
        if episode % 5 == 0 and episode > 20:
            img_loss = hybrid_agent.train_imagine_policy(trajectory, num_imagination_batches=5)
            if img_loss > 0:
                imagination_loss_episodes.append(episode)
                imagination_losses.append(img_loss)

        # 更新策略
        hybrid_agent.update(trajectory)

        # 测试
        if (episode + 1) % test_interval == 0:
            avg_reward, std_reward = test_policy(env, hybrid_agent, num_episodes=10, agent_type='hybrid')
            results['Hybrid Regret PPO']['episodes'].append(episode + 1)
            results['Hybrid Regret PPO']['rewards'].append(avg_reward)
            results['Hybrid Regret PPO']['stds'].append(std_reward)
            print(f"Episode {episode + 1:3d} | Test Reward: {avg_reward:.2f} ± {std_reward:.2f}")

    # 训练 Vanilla PPO
    print("\n[2/4] 训练 Vanilla PPO...")
    for episode in range(num_episodes):
        trajectory, _ = vanilla_ppo.collect_trajectory(env)
        vanilla_ppo.update(trajectory)

        if (episode + 1) % test_interval == 0:
            avg_reward, std_reward = test_policy(env, vanilla_ppo, num_episodes=10, agent_type='ppo')
            results['Vanilla PPO']['episodes'].append(episode + 1)
            results['Vanilla PPO']['rewards'].append(avg_reward)
            results['Vanilla PPO']['stds'].append(std_reward)
            print(f"Episode {episode + 1:3d} | Test Reward: {avg_reward:.2f} ± {std_reward:.2f}")

    # 训练 REINFORCE
    print("\n[3/4] 训练 REINFORCE...")
    for episode in range(num_episodes):
        trajectory, _ = reinforce_agent.collect_trajectory(env)
        reinforce_agent.update(trajectory)

        if (episode + 1) % test_interval == 0:
            avg_reward, std_reward = test_policy(env, reinforce_agent, num_episodes=10, agent_type='reinforce')
            results['REINFORCE']['episodes'].append(episode + 1)
            results['REINFORCE']['rewards'].append(avg_reward)
            results['REINFORCE']['stds'].append(std_reward)
            print(f"Episode {episode + 1:3d} | Test Reward: {avg_reward:.2f} ± {std_reward:.2f}")

    # 训练 A2C
    print("\n[4/4] 训练 A2C...")
    for episode in range(num_episodes):
        trajectory, _ = a2c_agent.collect_trajectory(env)
        a2c_agent.update(trajectory)

        if (episode + 1) % test_interval == 0:
            avg_reward, std_reward = test_policy(env, a2c_agent, num_episodes=10, agent_type='a2c')
            results['A2C']['episodes'].append(episode + 1)
            results['A2C']['rewards'].append(avg_reward)
            results['A2C']['stds'].append(std_reward)
            print(f"Episode {episode + 1:3d} | Test Reward: {avg_reward:.2f} ± {std_reward:.2f}")

    print("\n" + "=" * 80)
    print("训练完成！保存数据和生成图表...")
    print("=" * 80)

    # 初始化数据保存器
    data_saver = DataSaver(base_dir='data')

    # 保存配置
    config = {
        'num_episodes': num_episodes,
        'test_interval': test_interval,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'timestamp': data_saver.timestamp,
        'hybrid_ppo': {
            'use_imagination': True,
            'imagination_horizon': 15,
            'world_model_train_freq': 5
        }
    }
    data_saver.save_config(config)

    # 保存实验结果到Excel
    data_saver.save_results_to_excel(results)

    # 保存训练损失数据
    data_saver.save_training_losses(
        world_model_loss_episodes,
        world_model_losses,
        imagination_loss_episodes,
        imagination_losses
    )

    # 保存性能对比表
    data_saver.save_comparison_table(results)

    # 保存原始数据（便于后续分析）
    raw_data = {
        'results': results,
        'world_model_loss_episodes': world_model_loss_episodes,
        'world_model_losses': world_model_losses,
        'imagination_loss_episodes': imagination_loss_episodes,
        'imagination_losses': imagination_losses,
        'config': config
    }
    data_saver.save_raw_data(raw_data)

    # 绘制对比图（自动保存到data/figures）
    plot_selected_metrics(
        results,
        world_model_loss_episodes,
        world_model_losses,
        imagination_loss_episodes,
        imagination_losses,
        save_dir='data/figures'
    )

    # 打印最终结果
    print("\n" + "=" * 80)
    print("最终性能对比:")
    print("-" * 60)
    for algo_name, data in results.items():
        final_rewards = data['rewards'][-5:]
        avg_final = np.mean(final_rewards)
        std_final = np.std(final_rewards)
        print(f"{algo_name:20s}: {avg_final:.2f} ± {std_final:.2f}")
    print("-" * 60)
    print(f"\n所有数据已保存到 data/ 目录")
    print(f"时间戳: {data_saver.timestamp}")
    print("=" * 80)

    env.close()


if __name__ == "__main__":
    train_and_compare()

