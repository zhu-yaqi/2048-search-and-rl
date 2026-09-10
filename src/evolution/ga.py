"""用遗传算法(Genetic Algorithm)优化启发式评价函数的权重。

第 2 部分的启发式策略效果高度依赖各特征权重的取值，手工调参既费力又难达到
最优。本模块把"权重向量"作为优化对象（决策变量/基因型），把"该权重下贪心
策略的平均得分"作为适应度，用遗传算法自动搜索更优的权重组合。

进化算子：
- 选择：锦标赛选择(tournament)，兼顾选择压力与多样性；
- 交叉：算术混合交叉(BLX-α 思想的简化版)，在父代之间线性插值并加少量外推；
- 变异：高斯扰动，按一定概率对基因加噪声，提供探索能力；
- 精英保留：每代直接保留若干最优个体，保证不退化。

适应度评估用速度极快的一步前瞻贪心(GreedyAgent)，使得在 CPU 上也能在可接受
时间内完成多代进化；同一代内所有个体使用相同的随机种子集合，降低评估方差、
保证个体之间公平比较。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from ..heuristic.evaluation import DEFAULT_WEIGHTS, FEATURE_NAMES
from ..heuristic.search import GreedyAgent
from ..common.runner import play_game

N_GENES = len(FEATURE_NAMES)


@dataclass
class GAConfig:
    pop_size: int = 20
    n_generations: int = 15
    games_per_eval: int = 4        # 每个个体评估的对局数（取平均降方差）
    elite: int = 2                 # 精英个体数
    tournament_k: int = 3          # 锦标赛规模
    mutation_rate: float = 0.3     # 单基因变异概率
    mutation_sigma: float = 0.5    # 高斯变异标准差
    init_sigma: float = 1.5        # 初始种群在默认权重附近的散布
    seed: int = 0


@dataclass
class GAHistory:
    best_fitness: List[float] = field(default_factory=list)
    avg_fitness: List[float] = field(default_factory=list)
    best_weights: List[np.ndarray] = field(default_factory=list)


def fitness_of(weights: np.ndarray, n_games: int, base_seed: int) -> float:
    """适应度：给定权重下贪心策略的平均得分。"""
    agent = GreedyAgent(weights=weights)
    scores = [play_game(agent, seed=base_seed + i).score for i in range(n_games)]
    return float(np.mean(scores))


class GeneticOptimizer:
    def __init__(self, config: GAConfig = GAConfig()):
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)
        self.history = GAHistory()

    def _init_population(self) -> np.ndarray:
        """在默认权重附近随机初始化种群，保证起点合理同时具备多样性。"""
        base = DEFAULT_WEIGHTS
        pop = base + self.rng.normal(0, self.cfg.init_sigma, size=(self.cfg.pop_size, N_GENES))
        pop[0] = base.copy()  # 保留一个默认权重个体作为参照
        return pop

    def _evaluate(self, pop: np.ndarray, gen: int) -> np.ndarray:
        # 同一代使用相同种子集合，保证公平
        base_seed = self.cfg.seed + gen * 10007
        return np.array([
            fitness_of(ind, self.cfg.games_per_eval, base_seed) for ind in pop
        ])

    def _tournament(self, pop: np.ndarray, fits: np.ndarray) -> np.ndarray:
        idx = self.rng.integers(0, len(pop), size=self.cfg.tournament_k)
        best = idx[np.argmax(fits[idx])]
        return pop[best].copy()

    def _crossover(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        # 算术混合：alpha 在 [-0.25,1.25] 间取值，允许轻度外推
        alpha = self.rng.uniform(-0.25, 1.25, size=N_GENES)
        return alpha * p1 + (1 - alpha) * p2

    def _mutate(self, ind: np.ndarray) -> np.ndarray:
        mask = self.rng.random(N_GENES) < self.cfg.mutation_rate
        noise = self.rng.normal(0, self.cfg.mutation_sigma, size=N_GENES)
        ind = ind + mask * noise
        return ind

    def run(self, verbose: bool = True) -> Tuple[np.ndarray, GAHistory]:
        pop = self._init_population()
        best_overall, best_fit = None, -np.inf
        for gen in range(self.cfg.n_generations):
            fits = self._evaluate(pop, gen)
            order = np.argsort(fits)[::-1]
            pop, fits = pop[order], fits[order]

            if fits[0] > best_fit:
                best_fit, best_overall = fits[0], pop[0].copy()

            self.history.best_fitness.append(float(fits[0]))
            self.history.avg_fitness.append(float(fits.mean()))
            self.history.best_weights.append(pop[0].copy())
            if verbose:
                print(f"[Gen {gen + 1:02d}/{self.cfg.n_generations}] "
                      f"best={fits[0]:.0f} avg={fits.mean():.0f} "
                      f"weights={np.round(pop[0], 2)}", flush=True)

            # 生成下一代：精英保留 + 交叉变异
            new_pop = [pop[i].copy() for i in range(self.cfg.elite)]
            while len(new_pop) < self.cfg.pop_size:
                p1 = self._tournament(pop, fits)
                p2 = self._tournament(pop, fits)
                child = self._crossover(p1, p2)
                child = self._mutate(child)
                new_pop.append(child)
            pop = np.array(new_pop)

        return best_overall, self.history
