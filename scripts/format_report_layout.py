"""优化 report/实验报告.md 的排版：封面、YAML、图表编号、PDF 友好格式。"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "report" / "实验报告.md"

YAML = """---
title: "2048 游戏 AI 大作业实验报告"
subtitle: "采用人工智能方法实现并求解 2048 游戏"
author:
  - "郑敬仁（1820241067）"
  - "翟梓涵（1120240606）"
  - "朱雅琪（1120241676）"
date: "2026 年 6 月"
lang: zh-CN
documentclass: ctexart
papersize: a4
geometry: "margin=2.4cm"
fontsize: 12pt
linestretch: 1.35
toc: true
toc-depth: 3
numbersections: true
header-includes:
  - \\usepackage{booktabs}
  - \\usepackage{float}
  - \\usepackage{caption}
  - \\captionsetup{font=small,labelformat=simple,labelsep=quad}
  - \\setcounter{topnumber}{5}
  - \\renewcommand{\\figurename}{图}
  - \\renewcommand{\\tablename}{表}
---

"""

COVER = """# 2048 游戏 AI 大作业实验报告

| 项目 | 内容 |
| :-- | :-- |
| **课程** | 人工智能概论 |
| **题目** | 采用人工智能方法实现并求解 2048 游戏 |
| **指导老师** | 刘峡壁 |
| **团队成员** | 郑敬仁（1820241067）、翟梓涵（1120240606）、朱雅琪（1120241676） |
| **完成日期** | 2026 年 6 月 |
| **代码仓库** | `ai9000/`（含 `src/`、`scripts/`、`results/`） |

<div style="page-break-after: always;"></div>

"""

TLDR = """
## 核心结果速览

> 下表汇总各方法在多局评估下的主要指标（分数越高越好）。完整数据见第 8 章。

| 排名 | 方法 | 局数 | 平均分 | 最高分 | 平均最大块 | 典型最大块 | 速度 |
| :--: | :-- | :--: | --: | --: | --: | :-- | :-- |
| — | Random | 50 | 1,003 | 2,900 | 99 | 64–128 | 极快 |
| 4 | Greedy（默认） | 50 | 11,973 | 30,772 | 824 | 512–1024 | 很快 |
| 3 | Greedy（GA 进化） | 30 | 17,140 | 36,400 | 1,182 | 1024 | 很快 |
| **1** | **Expectimax (d=2)** | 12 | **38,403** | **81,456** | **2,304** | **2048** | 较慢 |
| 5 | Supervised PolicyNet | 100 | 5,313 | 12,428 | 388 | 256–512 | 快 |
| 6 | DQN（纯 Q，6k 步） | 20 | 2,461 | 3,304 | 208 | 128–256 | 快 |
| 2 | DQN + 安全护栏 | 20 | 9,090 | 14,620 | 653 | 512–1024 | 中 |

**结论**：Expectimax 最强；进化贪心次之；混合 RL（安全护栏）适合稳定演示；纯 DQN 在短训练下仍弱于启发式。

<div style="page-break-after: always;"></div>

"""

FIGURES = [
    ("../results/heuristic/greedy_hist.png", "图 8-1", "贪心策略 50 局分数分布"),
    ("../results/heuristic/greedy_tile.png", "图 8-2", "贪心策略最大方块分布"),
    ("../results/heuristic/expectimax_hist.png", "图 8-3", "Expectimax (depth=2) 分数分布"),
    ("../results/heuristic/expectimax_tile.png", "图 8-4", "Expectimax (depth=2) 最大方块分布"),
    ("../results/evolution/ga_curve.png", "图 8-5", "遗传算法进化曲线（每代最优 / 平均适应度）"),
    ("../results/supervised/policy_score_hist.png", "图 8-6", "监督学习网络 100 局分数分布"),
    ("../results/supervised/policy_tile_dist.png", "图 8-7", "监督学习网络最大方块分布"),
    ("../results/rl/dqn_score.png", "图 8-8", "DQN 训练过程平均得分（纯 Q 网络评估）"),
    ("../results/rl/dqn_tile.png", "图 8-9", "DQN 训练过程平均最大方块"),
    ("../results/rl/dqn_loss.png", "图 8-10", "DQN 训练过程 TD Loss"),
    ("../results/rl/dqn_pure_hist.png", "图 8-11", "纯 DQN 最终 20 局分数分布"),
    ("../results/rl/dqn_pure_tile.png", "图 8-12", "纯 DQN 最大方块分布"),
    ("../results/rl/dqn_hist.png", "图 8-13", "DQN + 安全护栏最终 20 局分数分布"),
    ("../results/rl/dqn_safe_tile.png", "图 8-14", "DQN + 安全护栏最大方块分布"),
]


def fig_block(path: str, label: str, caption: str) -> str:
    return (
        f"\n![{label} {caption}]({path}){{width=82%}}\n\n"
        f"*{label}　{caption}*\n"
    )


def apply_figure_format(text: str) -> str:
    """仅替换仍使用旧 alt 文本、且无 caption 的图片行（安全模式）。"""
    old_alts = {
        "../results/heuristic/greedy_hist.png": ("Greedy 分数分布", "图 8-1", "贪心策略 50 局分数分布"),
        "../results/heuristic/greedy_tile.png": ("Greedy 最大方块分布", "图 8-2", "贪心策略最大方块分布"),
        "../results/heuristic/expectimax_hist.png": ("Expectimax(d=2) 分数分布", "图 8-3", "Expectimax (depth=2) 分数分布"),
        "../results/heuristic/expectimax_tile.png": ("Expectimax(d=2) 最大方块分布", "图 8-4", "Expectimax (depth=2) 最大方块分布"),
        "../results/evolution/ga_curve.png": ("遗传算法进化曲线", "图 8-5", "遗传算法进化曲线（每代最优 / 平均适应度）"),
    }
    for path, (old_alt, label, caption) in old_alts.items():
        old_line = f"![{old_alt}]({path})"
        if old_line in text and f"*{label}" not in text:
            text = text.replace(old_line, fig_block(path, label, caption).strip())
    return text


def strip_old_header(text: str) -> str:
    # 去掉旧封面与 YAML
    text = re.sub(r"^---\n.*?\n---\n\n?", "", text, count=1, flags=re.DOTALL)
    text = re.sub(
        r"^# 《人工智能概论》课程实践大作业实验报告\n\n"
        r"\*\*题目：.*?\*\*\n\n"
        r"\*\*指导老师：.*?\*\*\n\n"
        r"\*\*团队成员：.*?\*\*\n\n"
        r"\*\*日期：.*?\*\*\n\n"
        r"---\n\n",
        "",
        text,
        count=1,
    )
    return text


def insert_tldr(text: str) -> str:
    if "## 核心结果速览" in text:
        return text
    return text.replace(
        "## 目录",
        TLDR.strip() + "\n\n## 目录",
        1,
    )


def add_table_captions(text: str) -> str:
    reps = [
        (
            "随机、贪心、Expectimax 在多局对局上的统计结果如下：\n\n| 策略",
            "随机、贪心、Expectimax 在多局对局上的统计结果如下（**表 8-1**）：\n\n**表 8-1　启发式策略性能对比**\n\n| 策略",
        ),
        (
            "**最大方块分布**：\n\n| 策略       | 64",
            "**表 8-2　启发式策略最大方块分布**\n\n| 策略       | 64",
        ),
        (
            "**最优权重对比**（与手工权重比较）：\n\n| 特征",
            "**表 8-3　进化前后启发式权重对比**\n\n| 特征",
        ),
        (
            "#### 评估对比\n\n| 策略",
            "#### 评估对比\n\n**表 8-4　默认权重 vs 进化权重贪心策略**\n\n| 策略",
        ),
        (
            "#### 训练指标\n\n| 指标",
            "#### 训练指标\n\n**表 8-5　监督学习验证集指标**\n\n| 指标",
        ),
        (
            "#### 对局评估（100 局",
            "#### 对局评估（100 局",
        ),
        (
            "| 指标           |        数值 |\n| -------------- | ----------: |\n| 平均分         | **5,313.2** |",
            "**表 8-6　监督学习网络对局评估（100 局）**\n\n| 指标           |        数值 |\n| -------------- | ----------: |\n| 平均分         | **5,313.2** |",
        ),
        (
            "| 评估步       |  750 |",
            "**表 8-7　DQN 训练过程各 checkpoint 指标**\n\n| 评估步       |  750 |",
        ),
        (
            "#### 8.4.3 最终评估\n\n| 配置",
            "#### 8.4.3 最终评估\n\n**表 8-8　纯 DQN vs 安全护栏最终对局对比（各 20 局）**\n\n| 配置",
        ),
        (
            "### 8.5 横向对比总结\n\n| 方法",
            "### 8.5 横向对比总结\n\n**表 8-9　全部方法横向对比**\n\n| 方法",
        ),
    ]
    for old, new in reps:
        text = text.replace(old, new)
    return text


def fix_section8_layout(text: str) -> str:
    # 实验章节引导语
    old = "> 复现方式：在项目根目录运行 `python scripts/run_all.py`。"
    new = (
        "> **说明**：本章给出各方法的定量结果与图表。复现命令见第 10 章；"
        "原始数据保存在 `results/summary.json`。\n\n"
        "> 复现方式：在项目根目录运行 `python scripts/run_all.py`。"
    )
    if old in text and "**说明**" not in text[:8000]:
        text = text.replace(old, new, 1)

    # 合并重复的图小节标题
    text = text.replace(
        "#### 贪心策略分数分布与最大方块分布\n\n",
        "",
    )
    text = text.replace(
        "#### Expectimax 分数分布与最大方块分布\n\n",
        "",
    )
    text = text.replace(
        "#### 进化曲线\n\n",
        "",
    )
    text = text.replace(
        "#### 对局分数分布\n\n",
        "",
    )
    text = text.replace(
        "#### 最大方块分布\n\n",
        "",
    )
    text = text.replace(
        "**纯 DQN 分数分布**：\n\n",
        "",
    )
    text = text.replace(
        "**DQN + 安全护栏分数分布**：\n\n",
        "",
    )
    return text


def fix_outdated(text: str) -> str:
    text = text.replace(
        "- **监督部分的训练曲线/分数分布图本次实验尚未生成**，已在报告中预留占位。",
        "- **监督 Loss/Accuracy 曲线未单独导出 PNG**：对局分布图已生成（图 8-6、图 8-7），训练指标见表 8-5。",
    )
    text = text.replace("平均分 5,353", "平均分 5,313")
    return text


def add_pdf_section(text: str) -> str:
    block = """
### 10.5 导出 PDF（推荐）

本报告使用 `$...$` / `$$...$$` 数学语法，配合 Pandoc + XeLaTeX 导出 PDF 时公式可正常渲染。

**环境**：安装 [Pandoc](https://pandoc.org/) 与 TeX 发行版（如 TeX Live / MiKTeX），并确保系统已安装中文字体（如 Microsoft YaHei）。

在项目根目录执行：

```bash
pandoc report/实验报告.md -o report/实验报告.pdf ^
  --pdf-engine=xelatex ^
  -V CJKmainfont="Microsoft YaHei" ^
  -V mainfont="Microsoft YaHei" ^
  --toc -N --dpi=300
```

Linux / macOS 将 `^` 换为 `\\` 继续行即可。若缺少 `ctexart`，可去掉 YAML 中的 `documentclass` 行，改用 `-V CJKmainfont=...` 单独指定字体。

"""
    if "### 10.5 导出 PDF" in text:
        return text
    return text.replace("---\n\n## 11. 附录", block + "---\n\n## 11. 附录", 1)


def main():
    text = REPORT.read_text(encoding="utf-8")
    text = strip_old_header(text)
    text = insert_tldr(text)
    text = apply_figure_format(text)
    text = add_table_captions(text)
    text = fix_section8_layout(text)
    text = fix_outdated(text)
    text = add_pdf_section(text)

    # 摘要后分页（PDF）
    text = text.replace(
        "## 摘要\n\n",
        "## 摘要\n\n",
    )
    if "<div style=\"page-break-after: always;\"></div>" not in text.split("## 目录")[0]:
        text = text.replace(
            "做了详细讨论。\n\n---\n\n## 目录",
            "做了详细讨论。\n\n<div style=\"page-break-after: always;\"></div>\n\n## 目录",
            1,
        )

    final = YAML + COVER + text
    REPORT.write_text(final, encoding="utf-8")
    print(f"排版已优化: {REPORT}")


if __name__ == "__main__":
    main()
