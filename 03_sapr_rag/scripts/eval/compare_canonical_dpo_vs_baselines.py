#!/usr/bin/env python3
"""对比 canonical-SFT→DPO 三数据集评测结果与 E14 canonical SFT / SFT+DPO 基线。

用法：
  python compare_canonical_dpo_vs_baselines.py --dpo_dir <eval_out> --output <md>
"""

import argparse
import json
from pathlib import Path

DATASETS = ["hotpotqa", "2wikimultihopqa", "musique"]

# 固定基线（来自 docs/experiment_tracker.md 已核验的 A 级全量结果）
# E14 = Canonical-answer SFT ckpt4150；SFT+DPO = E01（旧 SFT 起点 DPO）
BASELINES = {
    "hotpotqa": {
        "E14 canonical SFT": {"em": 0.4373, "f1": 0.5513, "cover_em": 0.4748},
        "SFT+DPO (E01)": {"em": 0.4008, "f1": 0.5233, "cover_em": 0.4693},
    },
    "2wikimultihopqa": {
        "E14 canonical SFT": {"em": 0.4051, "f1": 0.4513, "cover_em": 0.4188},
        "SFT+DPO (E01)": {"em": 0.3915, "f1": 0.4688, "cover_em": 0.4452},
    },
    "musique": {
        "E14 canonical SFT": {"em": 0.1651, "f1": 0.2405, "cover_em": 0.1841},
        "SFT+DPO (E01)": {"em": 0.1667, "f1": 0.2477, "cover_em": 0.2069},
    },
}


def load_metrics(dpo_dir: Path, ds: str):
    p = dpo_dir / ds / "metrics.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text())


def fmt(x):
    return f"{x:.4f}" if isinstance(x, (int, float)) else str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpo_dir", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    dpo_dir = Path(args.dpo_dir)
    lines = ["# Canonical-SFT→DPO vs E14 / SFT+DPO 三数据集对比", ""]

    for ds in DATASETS:
        m = load_metrics(dpo_dir, ds)
        lines.append(f"## {ds}")
        lines.append("")
        lines.append("| 方法 | EM | F1 | Cover-EM | avg_turns | max_turns_rate |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for name, b in BASELINES[ds].items():
            lines.append(
                f"| {name} | {fmt(b['em'])} | {fmt(b['f1'])} | {fmt(b['cover_em'])} | - | - |"
            )
        if m:
            lines.append(
                f"| **Canonical SFT+DPO (new)** | **{fmt(m['em'])}** | **{fmt(m['f1'])}** "
                f"| **{fmt(m['cover_em'])}** | {fmt(m.get('avg_turns'))} "
                f"| {fmt(m.get('max_turns_rate'))} |"
            )
            e14 = BASELINES[ds]["E14 canonical SFT"]
            lines.append("")
            lines.append(
                f"- 相对 E14：EM {m['em'] - e14['em']:+.4f} / "
                f"F1 {m['f1'] - e14['f1']:+.4f} / "
                f"Cover-EM {m['cover_em'] - e14['cover_em']:+.4f}"
            )
            sd = BASELINES[ds]["SFT+DPO (E01)"]
            lines.append(
                f"- 相对 SFT+DPO：EM {m['em'] - sd['em']:+.4f} / "
                f"F1 {m['f1'] - sd['f1']:+.4f} / "
                f"Cover-EM {m['cover_em'] - sd['cover_em']:+.4f}"
            )
        else:
            lines.append("| **Canonical SFT+DPO (new)** | 缺失 | 缺失 | 缺失 | - | - |")
        lines.append("")

    out = Path(args.output)
    out.write_text("\n".join(lines))
    print(f"[compare] wrote {out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
