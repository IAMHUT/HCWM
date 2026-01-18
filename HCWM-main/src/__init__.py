"""
RL Algorithms Comparison
强化学习算法性能对比实验
"""

__version__ = "1.0.0"
__author__ = "IAMHUT"

from .agents.hybrid_ppo import EnhancedHybridRegretPPO
from .agents.vanilla_ppo import VanillaPPO
from .agents.reinforce import REINFORCE
from .agents.a2c import A2C

__all__ = [
    'EnhancedHybridRegretPPO',
    'VanillaPPO',
    'REINFORCE',
    'A2C',
]
