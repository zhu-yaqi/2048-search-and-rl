# 2048-search-and-rl

A comparative study of search, evolutionary algorithms, supervised learning and deep reinforcement learning on 2048.

本项目用多种人工智能方法实现并求解 2048 游戏，覆盖大作业要求的全部 5 个部分：

1. **游戏系统**：完整的 4×4 2048 规则引擎 + 图形界面（可人玩 / 看 AI 演示）。
2. **启发式搜索决策**：可解释的特征评价函数 + 一步前瞻贪心 + 期望最大化（Expectimax）搜索。
3. **进化计算优化**：用遗传算法自动搜索评价函数的最优权重。
4. **监督学习**：用 Expectimax 专家自我对弈采集数据，训练卷积策略网络模仿专家。
5. **强化学习**：Dueling Double DQN + Prioritized Replay + n-step TD + DQfD 专家预训练，智能体与 2048 环境持续交互自我提升。

详细的方案原理、模型设计、技术分析与实验结果见 `report/实验报告.md`。

## 目录结构

```
.
├── README.md
├── requirements.txt
├── report/
│   └── 实验报告.md           # 实验报告（方案/原理/模型/结果分析）
├── src/
│   ├── game/                 # 游戏核心
│   │   ├── board.py          # 规则引擎(数值棋盘, 清晰版)
│   │   ├── fast.py           # 高性能引擎(指数表示+行查找表)
│   │   └── gui.py            # tkinter 图形界面
│   ├── heuristic/            # 第2部分
│   │   ├── evaluation.py     # 启发式特征与评价函数
│   │   └── search.py         # 贪心 / Expectimax
│   ├── evolution/            # 第3部分
│   │   └── ga.py             # 遗传算法
│   ├── supervised/           # 第4部分
│   │   ├── model.py          # 策略网络 + 网络智能体
│   │   ├── generate_data.py  # 专家数据采集
│   │   └── train.py          # 监督训练
│   ├── rl/                   # 第5部分
│   │   ├── env.py            # 强化学习环境
│   │   ├── features.py       # RL 状态特征与势函数
│   │   └── dqn.py            # Dueling DDQN / PER / n-step / DQfD
│   └── common/               # 通用工具
│       ├── encoding.py       # 棋盘 one-hot 编码
│       ├── runner.py         # 对局与多局评估
│       └── plotting.py       # 绘图
├── scripts/                  # 运行入口
│   ├── play_gui.py           # 启动图形界面
│   ├── run_heuristic.py      # 评估启发式策略
│   ├── run_evolution.py      # 运行进化优化
│   ├── run_supervised.py     # 监督学习
│   ├── run_rl.py             # 强化学习
│   └── run_all.py            # 一键跑全流程并产出结果
└── results/                  # 运行后生成的模型/图表/指标汇总
```

## 环境与安装

- Python ≥ 3.10（开发环境 3.12）
- 依赖：`numpy`、`torch`(CPU 即可)、`matplotlib`；`tkinter` 为标准库自带。

```bash
pip install -r requirements.txt
```

## 快速开始

```bash
# 1. 人类游玩（方向键控制）
python scripts/play_gui.py

# 2. 观看启发式 AI 自动演示
python scripts/play_gui.py --agent expectimax --depth 2
python scripts/play_gui.py --agent greedy

# 3. 评估启发式策略（贪心 + Expectimax）
python scripts/run_heuristic.py --games 30

# 4. 进化优化权重
python scripts/run_evolution.py --pop 16 --gens 12

# 5. 监督学习（生成专家数据并训练网络）
python scripts/run_supervised.py --samples 15000 --epochs 18

# 6. 强化学习
# 快速检查训练流程
python scripts/run_rl.py --steps 1000 --eval-every 200 --eval-games 3 --final-eval-games 5

# 正式训练：从零训练 Double DQN
python scripts/run_rl.py --steps 60000

# 正式训练：从监督学习模型热启动 Double DQN
python scripts/run_rl.py --steps 60000 --init results/supervised/policy.pt

# 7. 一键复现全部实验（结果写入 results/summary.txt）
python scripts/run_all.py
```

运行 `run_all.py` 后，可用训练好的网络观看演示：

```bash
python scripts/play_gui.py --agent net --model results/supervised/policy.pt
```

## 设计要点

- **强化学习优化**：RL 部分默认使用特征版 Dueling Double DQN，并加入优先经验回放、3-step TD、potential-based reward shaping、DQfD 风格专家 demonstration 预训练；最终还输出“无 Expectimax”与“Expectimax 安全护栏”两种评估。
- **解耦**：游戏规则、评价函数、搜索、学习算法各自独立，便于复用与替换。
- **高性能引擎**：把方块用指数表示、把一行编码成 16 位整数，离线枚举所有
  65536 种行的"向左滑动+合并"结果并查表，使 Expectimax 深层搜索、进化适应度
  评估、强化学习海量 rollout 都能在纯 CPU 上以可接受的速度运行。
- **统一评估**：所有策略都通过 `evaluate_agent` 多局评估，用平均分、最大方块
  分布等同一套指标横向比较，保证公平。
