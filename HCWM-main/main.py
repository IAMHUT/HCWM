"""
快速启动脚本
"""

from experiments.compare import train_and_compare

if __name__ == "__main__":
    print("=" * 80)
    print(" " * 20 + "强化学习算法对比实验")
    print(" " * 15 + "HCWM世界模型 + 想象轨迹训练版本 + 反事实后悔计算模块")
    print("=" * 80)
    train_and_compare()
