"""
Orchestrator مرحله ۵ — اجرا و مقایسه ۳ baseline تخصیص.

اجرا:
    python -m src.baselines.run_baselines
    python -m src.baselines.run_baselines --strategies greedy hungarian
    python -m src.baselines.run_baselines --debug

این اسکریپت:
    ۱) پیکربندی کالیبره‌شده (abm_calibrated.yaml) و overrides کالیبراسیون را بارگذاری می‌کند
    ۲) برای هر استراتژی (random, greedy, hungarian):
       - با dispatcher_strategy(name) آن را patch می‌کند
       - با run_multi_seed بر روی ۱۰ seed اجرا می‌کند (همان seedهای کالیبراسیون نهایی)
       - KPIها را جمع‌آوری می‌کند
    ۳) جدول مقایسه را چاپ + ذخیره می‌کند (CSV + TXT)
    ۴) دو نمودار publication-ready تولید می‌کند:
       - kpi_comparison.png  — bar chart 3 استراتژی + Target × ۶ KPI
       - kpi_distribution.png — boxplot توزیع روی ۱۰ seed برای هر استراتژی
    ۵) خلاصه JSON ذخیره می‌کند
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["savefig.bbox"] = "tight"


# ============================================================================
# تنظیمات گزارش
# ============================================================================

STRATEGIES: List[str] = ["random", "greedy", "hungarian", "hungarian_resolve"]
STRATEGY_LABEL_FA: Dict[str, str] = {
    "random": "تصادفی",
    "greedy": "حریصانه",
    "hungarian": "مجاری (batch)",
    "hungarian_resolve": "مجاری با حل مجدد",
}
STRATEGY_COLOR: Dict[str, str] = {
    "random": "#E45756",
    "greedy": "#54A24B",
    "hungarian": "#F58518",
    "hungarian_resolve": "#B279A2",
    "target": "#4C78A8",
}

# همان ۱۰ seed مرحله ۴ (consistency بین کالیبراسیون و baseline)
EVAL_SEEDS: List[int] = [42, 123, 456, 789, 1010, 1337, 2024, 3141, 5926, 7777]

KPI_REPORT: List[str] = [
    "CR", "mean_WT_min", "mean_DU",
    "cancel_rate", "no_driver_rate", "acceptance_rate",
]
KPI_LABEL_FA: Dict[str, str] = {
    "CR": "نرخ تکمیل (CR)",
    "mean_WT_min": "زمان انتظار (دقیقه)",
    "mean_DU": "بهره‌وری راننده (DU)",
    "cancel_rate": "نرخ لغو",
    "no_driver_rate": "نرخ بی‌راننده",
    "acceptance_rate": "نرخ پذیرش",
}
TARGET_KEY_OF: Dict[str, str] = {
    "CR": "CR_baseline",
    "mean_WT_min": "WT_baseline",
    "mean_DU": "DU_baseline",
    "cancel_rate": "cancel_rate",
    "no_driver_rate": "no_driver_rate",
    "acceptance_rate": "acceptance_rate",
}


# ============================================================================
# اجرای یک baseline
# ============================================================================

def evaluate_strategy(
    strategy_name: str,
    base_config: Dict[str, Any],
    targets: Dict[str, Any],
    zones_data: Dict[str, Any],
    overrides: Dict[str, Any],
    seeds: List[int],
) -> Dict[str, Any]:
    """
    اجرای یک استراتژی روی چند seed و گردآوری نتیجه.

    خروجی:
        dict شامل 'mean', 'std', 'per_seed', 'seeds', 'n_seeds', 'strategy'
    """
    from src.calibration.multi_seed_runner import run_multi_seed
    from src.baselines.dispatch_strategies import dispatcher_strategy

    logger.info("=" * 60)
    logger.info("Evaluating baseline: %s", strategy_name.upper())
    logger.info("=" * 60)

    # نکته معماری: dispatcher_strategy ابتدا dispatch_step را با baseline ما
    # جایگزین می‌کند. سپس run_multi_seed در داخل خود abm_param_overrides را
    # اعمال می‌کند که no_driver_threshold را به‌صورت wrapper روی baseline ما
    # می‌پیچد. در پایان، هر دو لایه به‌درستی restore می‌شوند.
    with dispatcher_strategy(strategy_name):
        result = run_multi_seed(
            base_config=base_config,
            targets=targets,
            zones_data=zones_data,
            overrides=dict(overrides),  # کپی برای ایمنی
            seeds=seeds,
            verbose=True,
        )

    result["strategy"] = strategy_name
    return result


# ============================================================================
# ساخت DataFrame و جدول مقایسه
# ============================================================================

def build_comparison_dataframe(
    results: Dict[str, Dict[str, Any]],
    targets: Dict[str, Any],
) -> pd.DataFrame:
    """ساخت DataFrame مقایسه از نتایج چند استراتژی + ردیف Target."""
    rows = []
    for strat, result in results.items():
        m = result["mean"]
        s = result["std"]
        row = {"strategy": strat}
        for kpi in KPI_REPORT:
            row[f"{kpi}_mean"] = m.get(kpi, float("nan"))
            row[f"{kpi}_std"] = s.get(kpi, 0.0)
        row["total_assignments"] = m.get("total_assignments", float("nan"))
        row["total_rejections"] = m.get("total_rejections", float("nan"))
        rows.append(row)
    df = pd.DataFrame(rows)

    # افزودن ردیف Target (برای رفرنس بصری در جدول)
    target_row = {"strategy": "target"}
    for kpi in KPI_REPORT:
        target_row[f"{kpi}_mean"] = float(targets.get(TARGET_KEY_OF[kpi], float("nan")))
        target_row[f"{kpi}_std"] = 0.0
    target_row["total_assignments"] = float("nan")
    target_row["total_rejections"] = float("nan")
    df = pd.concat([df, pd.DataFrame([target_row])], ignore_index=True)
    return df


def print_comparison_table(df: pd.DataFrame) -> str:
    """چاپ جدول متنی مقایسه (ready for thesis)."""
    lines: List[str] = []
    border = "─" * 100
    lines.append(border)
    lines.append("Baseline Comparison (10 seeds each, calibrated config)")
    lines.append(border)
    lines.append(
        f"{'Algorithm':<14}{'CR':>10}{'WT':>10}{'DU':>10}"
        f"{'cancel':>10}{'no_drv':>10}{'accept':>10}{'assigns':>11}"
    )
    lines.append(border)
    for _, row in df.iterrows():
        strat = str(row["strategy"])
        assigns = row["total_assignments"]
        assigns_str = f"{assigns:>11.0f}" if not pd.isna(assigns) else f"{'—':>11}"
        lines.append(
            f"{strat:<14}"
            f"{row['CR_mean']:>10.3f}"
            f"{row['mean_WT_min_mean']:>10.3f}"
            f"{row['mean_DU_mean']:>10.3f}"
            f"{row['cancel_rate_mean']:>10.3f}"
            f"{row['no_driver_rate_mean']:>10.3f}"
            f"{row['acceptance_rate_mean']:>10.3f}"
            + assigns_str
        )
    lines.append(border)
    text = "\n".join(lines)
    print(text)
    return text


# ============================================================================
# نمودارها
# ============================================================================

def plot_kpi_comparison(
    df: pd.DataFrame,
    targets: Dict[str, Any],
    out_path: Path,
) -> None:
    """نمودار میله‌ای ۳ استراتژی + Target × ۶ KPI (با خطای استاندارد)."""
    strategies = [s for s in df["strategy"].tolist() if s != "target"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), squeeze=False)

    for k, kpi in enumerate(KPI_REPORT):
        ax = axes[k // 3][k % 3]
        labels: List[str] = []
        values: List[float] = []
        errs: List[float] = []
        bar_colors: List[str] = []

        for strat in strategies:
            row = df[df["strategy"] == strat].iloc[0]
            labels.append(strat)
            values.append(float(row[f"{kpi}_mean"]))
            errs.append(float(row[f"{kpi}_std"]))
            bar_colors.append(STRATEGY_COLOR.get(strat, "gray"))

        # ستون target
        labels.append("target")
        values.append(float(targets.get(TARGET_KEY_OF[kpi], 0.0)))
        errs.append(0.0)
        bar_colors.append(STRATEGY_COLOR["target"])

        x = np.arange(len(labels))
        ax.bar(x, values, yerr=errs, color=bar_colors, edgecolor="white",
               capsize=4, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, fontsize=9)
        ax.set_title(KPI_LABEL_FA[kpi], fontsize=10)
        ax.grid(alpha=0.3, axis="y")
        for xi, v in zip(x, values):
            ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("مقایسه baselineها روی ۱۰ seed (calibrated config)", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("saved %s", out_path)


def plot_kpi_distribution(
    results: Dict[str, Dict[str, Any]],
    targets: Dict[str, Any],
    out_path: Path,
) -> None:
    """Boxplot توزیع KPIها روی ۱۰ seed برای هر استراتژی + خط هدف."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), squeeze=False)
    strategies = list(results.keys())

    for k, kpi in enumerate(KPI_REPORT):
        ax = axes[k // 3][k % 3]
        data: List[List[float]] = []
        labels: List[str] = []
        for strat in strategies:
            per_seed = results[strat]["per_seed"]
            vals = [float(d.get(kpi, 0.0)) for d in per_seed]
            data.append(vals)
            labels.append(strat)
        ax.boxplot(data, tick_labels=labels, showmeans=True)
        ax.set_title(KPI_LABEL_FA[kpi], fontsize=10)
        ax.grid(alpha=0.3, axis="y")
        # خط افقی target
        tgt = float(targets.get(TARGET_KEY_OF[kpi], 0.0))
        ax.axhline(tgt, color="red", linestyle="--", linewidth=1.2, alpha=0.7,
                   label=f"target={tgt:.3f}")
        ax.legend(fontsize=7, loc="best")

    fig.suptitle("توزیع KPIها روی ۱۰ seed (با خط هدف کالیبراسیون)", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("saved %s", out_path)


# ============================================================================
# CLI و main
# ============================================================================

def parse_args() -> argparse.Namespace:
    """تجزیه آرگومان‌های خط فرمان."""
    p = argparse.ArgumentParser(description="Evaluate dispatch baselines.")
    p.add_argument("--config", type=str,
                   default="experiments/configs/abm_calibrated.yaml")
    p.add_argument("--targets", type=str,
                   default="data/calibration/targets.json")
    p.add_argument("--zones", type=str, default="data/calibration/zones.json")
    p.add_argument("--out-dir", type=str, default="experiments/results")
    p.add_argument("--figures-dir", type=str, default="figures/baselines")
    p.add_argument("--strategies", nargs="+", default=STRATEGIES,
                   choices=STRATEGIES,
                   help="Subset of strategies to evaluate (default: all 3).")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def extract_calibration_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    استخراج overrides رفتاری از calibration block در abm_calibrated.yaml.

    این overrides شامل p_cancel_per_step, patience_scale, beta_a, beta_b,
    no_driver_threshold_steps است که run_multi_seed آن‌ها را به abm_param_overrides
    پاس می‌دهد. n_drivers و search_radius_km هم در config top-level هستند
    که بدون مشکل دوباره اعمال می‌شوند.
    """
    cal = config.get("calibration", {})
    return dict(cal.get("overrides", {}))


def main() -> int:
    args = parse_args()
    from src.abm.utils import load_config, load_targets, load_zones, setup_logging
    setup_logging(level=logging.DEBUG if args.debug else logging.INFO)

    root = Path(__file__).resolve().parents[2]
    config = load_config(root / args.config)
    targets = load_targets(root / args.targets)
    zones_data = load_zones(root / args.zones)

    out_dir = root / args.out_dir
    figures_dir = root / args.figures_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # overrides از calibration block (همان پارامترهای مرحله ۴)
    overrides = extract_calibration_overrides(config)
    logger.info("Calibration overrides: %s", overrides)

    # اجرای هر استراتژی
    results: Dict[str, Dict[str, Any]] = {}
    for strat in args.strategies:
        results[strat] = evaluate_strategy(
            strategy_name=strat,
            base_config=config,
            targets=targets,
            zones_data=zones_data,
            overrides=overrides,
            seeds=EVAL_SEEDS,
        )

    # ساخت جدول مقایسه و ذخیره CSV
    df = build_comparison_dataframe(results, targets)
    csv_path = out_dir / "baselines_comparison.csv"
    df.to_csv(csv_path, index=False)
    logger.info("saved %s", csv_path)

    # CSV per-seed (برای تحلیل آماری دقیق‌تر)
    per_seed_rows = []
    for strat, result in results.items():
        for seed_idx, per_seed in zip(result["seeds"], result["per_seed"]):
            row = {"strategy": strat, "seed": seed_idx}
            row.update({k: v for k, v in per_seed.items() if isinstance(v, (int, float))})
            per_seed_rows.append(row)
    per_seed_df = pd.DataFrame(per_seed_rows)
    per_seed_csv = out_dir / "baselines_per_seed.csv"
    per_seed_df.to_csv(per_seed_csv, index=False)
    logger.info("saved %s", per_seed_csv)

    # جدول مقایسه (متن)
    table_text = print_comparison_table(df)
    table_path = out_dir / "baselines_comparison_table.txt"
    table_path.write_text(table_text, encoding="utf-8")
    logger.info("saved %s", table_path)

    # نمودارها
    plot_kpi_comparison(df, targets, figures_dir / "kpi_comparison.png")
    plot_kpi_distribution(results, targets, figures_dir / "kpi_distribution.png")

    # خلاصه JSON
    summary = {
        strat: {
            "mean": result["mean"],
            "std": result["std"],
            "n_seeds": result["n_seeds"],
            "seeds": result["seeds"],
        }
        for strat, result in results.items()
    }
    summary_path = out_dir / "baselines_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("saved %s", summary_path)

    print("\n" + "=" * 60)
    print("BASELINES EVALUATION COMPLETE")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
