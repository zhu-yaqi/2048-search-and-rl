"""从 results/summary.json 中已有的统计数据**重新生成**各阶段图表。

为什么需要这个脚本：
  - 旧版 plotting.py 因 matplotlib 字体配置 bug，部分图存在中文方块；
  - 直接重跑 run_heuristic / run_evolution 要花数十分钟，但底层 scores 已经在
    summary.json 里完整保存了。读出来重新画即可，几秒完成。

用法：
    python scripts/regenerate_plots.py
"""
import json
import os

import _bootstrap  # noqa: F401

from src.common.plotting import (
    plot_curves,
    plot_score_hist,
    plot_tile_distribution,
    _ACTIVE_FONT,
)


SUMMARY_PATH = os.path.join("results", "summary.json")


def main():
    print(f"使用中文字体: {_ACTIVE_FONT}")

    if not os.path.exists(SUMMARY_PATH):
        raise FileNotFoundError(f"找不到 {SUMMARY_PATH}，请先跑 run_all.py")
    with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ---------- 启发式 ----------
    heu = data.get("heuristic", {})
    if "greedy" in heu and heu["greedy"].get("scores"):
        plot_score_hist(heu["greedy"]["scores"], "贪心策略 50 局分数分布",
                        "results/heuristic/greedy_hist.png", bins=15)
        if heu["greedy"].get("tile_distribution"):
            plot_tile_distribution(heu["greedy"]["tile_distribution"],
                                   "贪心策略最大方块分布",
                                   "results/heuristic/greedy_tile.png")
    if "expectimax_d2" in heu and heu["expectimax_d2"].get("scores"):
        plot_score_hist(heu["expectimax_d2"]["scores"],
                        "Expectimax(depth=2) 分数分布",
                        "results/heuristic/expectimax_hist.png", bins=10)
        if heu["expectimax_d2"].get("tile_distribution"):
            plot_tile_distribution(heu["expectimax_d2"]["tile_distribution"],
                                   "Expectimax(depth=2) 最大方块分布",
                                   "results/heuristic/expectimax_tile.png")
    if "random" in heu and heu["random"].get("scores"):
        plot_score_hist(heu["random"]["scores"], "随机策略 50 局分数分布",
                        "results/heuristic/random_hist.png", bins=15)

    # ---------- 进化 ----------
    evo = data.get("evolution", {})
    if evo.get("best_fitness_curve") and evo.get("avg_fitness_curve"):
        gens = list(range(1, len(evo["best_fitness_curve"]) + 1))
        plot_curves(gens, {
            "每代最优": evo["best_fitness_curve"],
            "每代平均": evo["avg_fitness_curve"],
        }, "遗传算法进化曲线", "代数", "适应度(贪心平均得分)",
            "results/evolution/ga_curve.png")
    if evo.get("greedy_evolved", {}).get("scores"):
        plot_score_hist(evo["greedy_evolved"]["scores"],
                        "贪心(进化权重) 分数分布",
                        "results/evolution/greedy_evolved_hist.png", bins=15)
    if evo.get("greedy_default", {}).get("scores"):
        plot_score_hist(evo["greedy_default"]["scores"],
                        "贪心(默认权重) 分数分布",
                        "results/evolution/greedy_default_hist.png", bins=15)

    # ---------- 监督学习 ----------
    sup = data.get("supervised", {})
    sup_scores = sup.get("eval", {}).get("scores")
    if sup_scores:
        plot_score_hist(sup_scores, "监督学习网络对局分数分布",
                        "results/supervised/policy_score_hist.png", bins=15)
    sup_tile = sup.get("eval", {}).get("tile_distribution")
    if sup_tile:
        plot_tile_distribution(sup_tile, "监督学习网络最大方块分布",
                               "results/supervised/policy_tile_dist.png")

    # ---------- 强化学习 ----------
    rl = data.get("rl", {})
    if rl.get("eval_steps") and rl.get("eval_avg_score"):
        kind = "DQN+安全护栏" if rl.get("config", {}).get(
            "eval_use_safe_policy_during_train", False) else "纯 DQN"
        plot_curves(rl["eval_steps"], {"平均得分": rl["eval_avg_score"]},
                    f"DQN 训练过程平均得分（评估={kind}）",
                    "环境步数", "平均得分",
                    "results/rl/dqn_score.png")
        plot_curves(rl["eval_steps"], {"平均最大方块": rl["eval_avg_tile"]},
                    f"DQN 训练过程平均最大方块（评估={kind}）",
                    "环境步数", "最大方块",
                    "results/rl/dqn_tile.png")
        if rl.get("loss_curve"):
            plot_curves(rl["eval_steps"], {"TD Loss": rl["loss_curve"]},
                        "DQN 训练过程 TD Loss",
                        "环境步数", "Huber Loss",
                        "results/rl/dqn_loss.png")
    # 兼容旧格式 final_eval
    final_keys = [("final_eval_pure_dqn", "DQN(纯 Q 网络) 最终分数分布",
                   "results/rl/dqn_pure_hist.png",
                   "DQN(纯 Q 网络) 最大方块分布",
                   "results/rl/dqn_pure_tile.png"),
                  ("final_eval_dqn_safe", "DQN+安全护栏 最终分数分布",
                   "results/rl/dqn_hist.png",
                   "DQN+安全护栏 最大方块分布",
                   "results/rl/dqn_safe_tile.png"),
                  ("final_eval", "DQN 最终分数分布",
                   "results/rl/dqn_hist.png", None, None)]
    for key, h_title, h_path, t_title, t_path in final_keys:
        sec = rl.get(key)
        if not sec:
            continue
        if sec.get("scores"):
            plot_score_hist(sec["scores"], h_title, h_path, bins=15)
        if t_title and sec.get("tile_distribution"):
            plot_tile_distribution(sec["tile_distribution"], t_title, t_path)

    print("\n所有可重生成的图表已刷新。")


if __name__ == "__main__":
    main()
