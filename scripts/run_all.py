import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run_cmd(name: str, cmd: list[str], summary_lines: list[str]) -> bool:
    print("\n" + "=" * 80)
    print(f"[run_all] {name}")
    print("[cmd]", " ".join(cmd))
    print("=" * 80, flush=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)

    ret = subprocess.run(cmd, cwd=ROOT, env=env)

    ok = ret.returncode == 0
    status = "OK" if ok else f"FAILED({ret.returncode})"
    summary_lines.append(f"[{status}] {name}: {' '.join(cmd)}")

    if not ok:
        print(f"[run_all] {name} 失败，继续执行后续步骤。", flush=True)

    return ok


def read_supervised_summary(summary_lines: list[str]):
    p = ROOT / "results" / "supervised" / "summary.json"
    if not p.exists():
        summary_lines.append("[info] supervised summary.json not found")
        return

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        ev = data.get("eval", {})
        summary_lines.append("")
        summary_lines.append("[supervised final]")
        summary_lines.append(f"model: {data.get('model')}")
        summary_lines.append(f"data: {data.get('data')}")
        summary_lines.append(f"best_val_acc: {data.get('best_val_acc')}")
        summary_lines.append(f"avg_score: {ev.get('avg_score')}")
        summary_lines.append(f"median_score: {ev.get('median_score')}")
        summary_lines.append(f"max_score: {ev.get('max_score')}")
        summary_lines.append(f"rate_512: {ev.get('rate_512')}")
        summary_lines.append(f"rate_1024: {ev.get('rate_1024')}")
        summary_lines.append(f"rate_2048: {ev.get('rate_2048')}")
    except Exception as e:
        summary_lines.append(f"[warn] failed to read supervised summary: {e}")


def main():
    parser = argparse.ArgumentParser(description="一键运行 2048 项目实验流程")

    parser.add_argument("--quick", action="store_true", help="快速测试模式，缩小所有实验规模")
    parser.add_argument("--skip-heuristic", action="store_true")
    parser.add_argument("--skip-evolution", action="store_true")
    parser.add_argument("--skip-supervised", action="store_true")
    parser.add_argument("--skip-rl", action="store_true")

    parser.add_argument(
        "--regen-supervised-data",
        action="store_true",
        help="强制重新生成监督学习专家数据；默认如果 results/supervised/data.npz 存在则复用",
    )

    parser.add_argument("--heuristic-games", type=int, default=None)
    parser.add_argument("--evo-pop", type=int, default=None)
    parser.add_argument("--evo-gens", type=int, default=None)
    parser.add_argument("--sup-samples", type=int, default=None)
    parser.add_argument("--sup-epochs", type=int, default=None)
    parser.add_argument("--sup-eval-games", type=int, default=None)
    parser.add_argument("--rl-steps", type=int, default=None)

    args = parser.parse_args()

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    if args.quick:
        heuristic_games = args.heuristic_games or 5
        evo_pop = args.evo_pop or 4
        evo_gens = args.evo_gens or 2
        sup_samples = args.sup_samples or 15000
        sup_epochs = args.sup_epochs or 5
        sup_eval_games = args.sup_eval_games or 20
        rl_steps = args.rl_steps or 5000
    else:
        heuristic_games = args.heuristic_games or 30
        evo_pop = args.evo_pop or 16
        evo_gens = args.evo_gens or 12
        sup_samples = args.sup_samples or 80000
        sup_epochs = args.sup_epochs or 25
        sup_eval_games = args.sup_eval_games or 100
        rl_steps = args.rl_steps or 60000

    summary_lines = []
    summary_lines.append("2048 experiment summary")
    summary_lines.append(f"time: {datetime.now().isoformat(timespec='seconds')}")
    summary_lines.append(f"quick: {args.quick}")
    summary_lines.append("")

    if not args.skip_heuristic:
        run_cmd(
            "heuristic evaluation",
            [
                PYTHON,
                "scripts/run_heuristic.py",
                "--games",
                str(heuristic_games),
            ],
            summary_lines,
        )

    if not args.skip_evolution:
        run_cmd(
            "evolution optimization",
            [
                PYTHON,
                "scripts/run_evolution.py",
                "--pop",
                str(evo_pop),
                "--gens",
                str(evo_gens),
            ],
            summary_lines,
        )

    if not args.skip_supervised:
        supervised_data = ROOT / "results" / "supervised" / "data.npz"

        if supervised_data.exists() and not args.regen_supervised_data:
            sup_cmd = [
                PYTHON,
                "scripts/run_supervised.py",
                "--reuse-data",
                "--epochs",
                str(sup_epochs),
                "--eval-games",
                str(sup_eval_games),
            ]
        else:
            sup_cmd = [
                PYTHON,
                "scripts/run_supervised.py",
                "--samples",
                str(sup_samples),
                "--epochs",
                str(sup_epochs),
                "--eval-games",
                str(sup_eval_games),
            ]

        run_cmd("supervised learning", sup_cmd, summary_lines)
        read_supervised_summary(summary_lines)

    if not args.skip_rl:
        init_model = ROOT / "results" / "supervised" / "policy.pt"

        if init_model.exists():
            run_cmd(
                "reinforcement learning",
                [
                    PYTHON,
                    "scripts/run_rl.py",
                    "--steps",
                    str(rl_steps),
                    "--init",
                    "results/supervised/policy.pt",
                ],
                summary_lines,
            )
        else:
            msg = "[skip] reinforcement learning: results/supervised/policy.pt not found"
            print(msg)
            summary_lines.append(msg)

    out = results_dir / "summary.txt"
    out.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("\n" + "=" * 80)
    print(f"[run_all] summary saved to {out}")
    print("=" * 80)
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
