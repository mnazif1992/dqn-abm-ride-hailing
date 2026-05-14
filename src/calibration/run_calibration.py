"""
اسکریپت اصلی کالیبراسیون ABM — Orchestrator مرحله ۴.

این اسکریپت تمام مراحل کالیبراسیون را به ترتیب اجرا می‌کند:
    ۱) اجرای ABM با config پیش‌فرض (baseline) و گزارش KPIها
    ۲) اجرای Grid Search روی فضای جستجوی پیش‌فرض (یا --skip-grid برای حذف)
    ۳) انتخاب و گزارش بهترین ترکیب
    ۴) اجرای مجدد ABM با config برنده برای تأیید (با ۱۰ seed)
    ۵) تولید سه نمودار مقایسه‌ای:
       - mape_distribution.png        توزیع MAPE در ترکیب‌ها
       - parameter_sensitivity.png    حساسیت MAPE به هر پارامتر
       - default_vs_calibrated.png    مقایسه ۶ KPI قبل/بعد
    ۶) چاپ جدول مقایسه «default vs calibrated» برای جدول ۴.۵ پایان‌نامه

نحوه اجرا:
    python -m src.calibration.run_calibration
    python -m src.calibration.run_calibration --skip-grid    # فقط نمودارها از CSV موجود
    python -m src.calibration.run_calibration --n-workers 4
    python -m src.calibration.run_calibration --results-csv experiments/logs/grid_search_v3_results.csv --skip-grid
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")   # back-end بدون نمایش (لازم برای headless)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# تنظیمات matplotlib برای فارسی-سازگار
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["savefig.bbox"] = "tight"

# لیست KPIهای ۶گانه‌ی گزارش
KPI_REPORT_ORDER: List[str] = [
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
# نمودارها
# ============================================================================

def plot_mape_distribution(
    df: pd.DataFrame,
    out_path: Path,
    threshold: float = 15.0,
) -> None:
    """نمودار توزیع MAPE وزن‌دار در همه ترکیب‌ها."""
    valid = df[df.get("error").isna()] if "error" in df.columns else df
    vals = valid["mape_weighted"].astype(float).dropna()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(vals, bins=30, color="#4C78A8", edgecolor="white", alpha=0.85)
    ax.axvline(threshold, color="red", linestyle="--", linewidth=1.5,
               label=f"threshold {threshold:.0f}%")
    if not vals.empty:
        best_mape = vals.min()
        ax.axvline(best_mape, color="green", linestyle="-", linewidth=1.5,
                   label=f"best {best_mape:.2f}%")
    ax.set_xlabel("MAPE وزن‌دار (%)")
    ax.set_ylabel("تعداد ترکیب‌ها")
    ax.set_title(f"توزیع MAPE وزن‌دار روی {len(vals)} ترکیب")
    ax.legend()
    ax.grid(alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("saved %s", out_path)


def plot_parameter_sensitivity(df: pd.DataFrame, out_path: Path) -> None:
    """نمودار حساسیت MAPE وزن‌دار به هر پارامتر."""
    valid = df[df.get("error").isna()] if "error" in df.columns else df
    param_cols = [c for c in valid.columns if c.startswith("param_")]
    if not param_cols:
        logger.warning("No param_* columns; cannot plot sensitivity.")
        return

    n = len(param_cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 3.2 * nrows), squeeze=False,
    )

    for i, col in enumerate(param_cols):
        ax = axes[i // ncols][i % ncols]
        groups = valid.groupby(col)["mape_weighted"].apply(list)
        labels = [str(k) for k in groups.index]
        data = list(groups.values)
        ax.boxplot(data, labels=labels, showmeans=True)
        ax.set_title(col[len("param_"):], fontsize=10)
        ax.set_ylabel("MAPE (%)")
        ax.grid(alpha=0.3)

    # حذف زیرنمودارهای اضافه
    for j in range(n, nrows * ncols):
        fig.delaxes(axes[j // ncols][j % ncols])

    fig.suptitle("حساسیت MAPE به پارامترهای جستجو", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("saved %s", out_path)


def plot_default_vs_calibrated(
    default_out: Dict[str, float],
    calibrated_out: Dict[str, float],
    targets: Dict[str, Any],
    out_path: Path,
) -> None:
    """نمودار میله‌ای مقایسه ۶ KPI: default vs calibrated vs target."""
    kpis = KPI_REPORT_ORDER
    labels = [KPI_LABEL_FA[k] for k in kpis]

    def_vals = [float(default_out.get(k, 0.0)) for k in kpis]
    cal_vals = [float(calibrated_out.get(k, 0.0)) for k in kpis]
    tgt_vals = [float(targets.get(TARGET_KEY_OF[k], 0.0)) for k in kpis]

    x = np.arange(len(kpis))
    width = 0.27
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width, def_vals, width, label="Default", color="#E45756")
    ax.bar(x, cal_vals, width, label="Calibrated", color="#54A24B")
    ax.bar(x + width, tgt_vals, width, label="Target", color="#4C78A8")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("مقدار")
    ax.set_title("مقایسه KPIها: Default vs Calibrated vs Target")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    # برچسب عددی روی میله‌ها
    for xi, v in zip(x - width, def_vals):
        ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    for xi, v in zip(x, cal_vals):
        ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    for xi, v in zip(x + width, tgt_vals):
        ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("saved %s", out_path)


# ============================================================================
# جدول مقایسه (برای جدول ۴.۵ پایان‌نامه)
# ============================================================================

def print_comparison_table(
    default_out: Dict[str, float],
    calibrated_out: Dict[str, float],
    targets: Dict[str, Any],
    default_validation: Dict[str, Any],
    calibrated_validation: Dict[str, Any],
) -> str:
    """چاپ جدول مقایسه default vs calibrated vs target."""
    lines: List[str] = []
    border = "─" * 90
    lines.append(border)
    lines.append("Comparison Table: Default vs Calibrated (جدول ۴.۵ پایان‌نامه)")
    lines.append(border)
    lines.append(
        f"{'KPI':<28}{'Target':>10}{'Default':>11}{'MAPE-D%':>10}"
        f"{'Calibrated':>13}{'MAPE-C%':>10}{'Δ':>10}"
    )
    lines.append(border)

    for kpi in KPI_REPORT_ORDER:
        tgt = float(targets.get(TARGET_KEY_OF[kpi], 0.0))
        d_val = float(default_out.get(kpi, 0.0))
        c_val = float(calibrated_out.get(kpi, 0.0))

        if kpi == "mean_WT_min":
            d_mape = default_validation["mape_per_kpi"].get("WT", float("nan"))
            c_mape = calibrated_validation["mape_per_kpi"].get("WT", float("nan"))
        elif kpi == "mean_DU":
            d_mape = default_validation["mape_per_kpi"].get("DU", float("nan"))
            c_mape = calibrated_validation["mape_per_kpi"].get("DU", float("nan"))
        elif kpi in ("CR", "cancel_rate", "no_driver_rate"):
            d_mape = default_validation["mape_per_kpi"].get(kpi, float("nan"))
            c_mape = calibrated_validation["mape_per_kpi"].get(kpi, float("nan"))
        else:  # acceptance_rate
            d_mape = abs(d_val - tgt) / max(abs(tgt), 1e-9) * 100.0
            c_mape = abs(c_val - tgt) / max(abs(tgt), 1e-9) * 100.0

        delta = c_val - d_val
        lines.append(
            f"{KPI_LABEL_FA[kpi]:<28}{tgt:>10.4f}{d_val:>11.4f}{d_mape:>10.2f}"
            f"{c_val:>13.4f}{c_mape:>10.2f}{delta:>+10.4f}"
        )

    lines.append(border)
    lines.append(
        f"MAPE weighted   default: {default_validation['mape_weighted']:.2f}%"
        f"   calibrated: {calibrated_validation['mape_weighted']:.2f}%"
    )
    lines.append(
        f"POM patterns    default: {default_validation['n_patterns_passed']}/5"
        f"   calibrated: {calibrated_validation['n_patterns_passed']}/5"
    )
    lines.append(border)
    text = "\n".join(lines)
    print(text)
    return text


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    """تجزیه آرگومان‌های خط فرمان."""
    p = argparse.ArgumentParser(description="Run full ABM calibration pipeline.")
    p.add_argument("--config", type=str,
                   default="experiments/configs/abm_default.yaml")
    p.add_argument("--targets", type=str,
                   default="data/calibration/targets.json")
    p.add_argument("--zones", type=str, default="data/calibration/zones.json")
    p.add_argument("--results-csv", type=str,
                   default="experiments/logs/grid_search_results.csv")
    p.add_argument("--calibrated-yaml", type=str,
                   default="experiments/configs/abm_calibrated.yaml")
    p.add_argument("--figures-dir", type=str, default="figures/calibration")
    p.add_argument("--results-dir", type=str, default="experiments/results")
    p.add_argument("--n-workers", type=int, default=None)
    p.add_argument("--skip-grid", action="store_true",
                   help="نادیده گرفتن Grid Search، فقط نمودارها از CSV موجود")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    from src.abm.utils import load_config, load_targets, load_zones, setup_logging
    setup_logging(level=logging.DEBUG if args.debug else logging.INFO)

    root = Path(__file__).resolve().parents[2]
    config = load_config(root / args.config)
    targets = load_targets(root / args.targets)
    zones_data = load_zones(root / args.zones)

    results_csv = root / args.results_csv
    calibrated_yaml = root / args.calibrated_yaml
    figures_dir = root / args.figures_dir
    results_dir = root / args.results_dir
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---- ۱) baseline default ----
    logger.info("=" * 60)
    logger.info("Step 1: running default ABM (multi-seed baseline)")
    logger.info("=" * 60)
    from src.calibration.multi_seed_runner import run_multi_seed
    from src.calibration.validate import validate_calibration, format_validation_report

    default_multi = run_multi_seed(
        base_config=config, targets=targets, zones_data=zones_data,
        overrides=None, seeds=[42, 123, 456], verbose=True,
    )
    default_out = default_multi["mean"]
    default_validation = validate_calibration(default_out, targets)
    print(format_validation_report(default_validation, header="Default ABM (uncalibrated)"))

    # ---- ۲) Grid Search ----
    if not args.skip_grid:
        logger.info("=" * 60)
        logger.info("Step 2: Grid Search")
        logger.info("=" * 60)
        from src.calibration.grid_search import run_grid_search
        df = run_grid_search(
            base_config=config, targets=targets, zones_data=zones_data,
            n_workers=args.n_workers,
            results_csv=results_csv, calibrated_yaml=calibrated_yaml,
            resume=True,
        )
    else:
        if not results_csv.exists():
            logger.error("--skip-grid اما %s وجود ندارد.", results_csv)
            return 1
        df = pd.read_csv(results_csv)
        logger.info("Loaded %d rows from %s", len(df), results_csv)
        if not calibrated_yaml.exists():
            from src.calibration.grid_search import select_best, save_calibrated_config
            best = select_best(df)
            save_calibrated_config(best, config, calibrated_yaml)

    # ---- ۳) اجرای config برنده با ۱۰ seed ----
    logger.info("=" * 60)
    logger.info("Step 3: re-running calibrated config with 10 seeds")
    logger.info("=" * 60)
    from src.calibration.grid_search import select_best
    valid = df[df.get("error").isna()] if "error" in df.columns else df
    best_row = select_best(valid)

    best_overrides: Dict[str, Any] = {}
    for k, v in best_row.items():
        if not k.startswith("param_"):
            continue
        param_name = k[len("param_"):]
        if isinstance(v, (np.integer,)):
            v = int(v)
        elif isinstance(v, (np.floating,)):
            v = float(v)
        best_overrides[param_name] = v

    calibrated_multi = run_multi_seed(
        base_config=config, targets=targets, zones_data=zones_data,
        overrides=best_overrides,
        seeds=[42, 123, 456, 789, 1010, 1337, 2024, 3141, 5926, 7777],
        verbose=True,
    )
    calibrated_out = calibrated_multi["mean"]
    calibrated_validation = validate_calibration(calibrated_out, targets)
    print(format_validation_report(calibrated_validation, header="Calibrated ABM (10 seeds)"))

    # ---- ۴) نمودارها ----
    logger.info("=" * 60)
    logger.info("Step 4: producing figures")
    logger.info("=" * 60)
    plot_mape_distribution(df, figures_dir / "mape_distribution.png")
    plot_parameter_sensitivity(df, figures_dir / "parameter_sensitivity.png")
    plot_default_vs_calibrated(
        default_out, calibrated_out, targets,
        figures_dir / "default_vs_calibrated.png",
    )

    # ---- ۵) جدول مقایسه ----
    logger.info("=" * 60)
    logger.info("Step 5: comparison table (for thesis Table 4.5)")
    logger.info("=" * 60)
    table_text = print_comparison_table(
        default_out, calibrated_out, targets,
        default_validation, calibrated_validation,
    )
    table_path = results_dir / "comparison_table.txt"
    table_path.write_text(table_text, encoding="utf-8")
    logger.info("saved %s", table_path)

    # ذخیره خلاصه JSON
    summary_path = results_dir / "calibration_summary.json"
    summary_data = {
        "default": {
            "mean": default_out, "std": default_multi["std"],
            "validation": {k: v for k, v in default_validation.items() if k != "details"},
        },
        "calibrated": {
            "mean": calibrated_out, "std": calibrated_multi["std"],
            "validation": {k: v for k, v in calibrated_validation.items() if k != "details"},
            "best_overrides": best_overrides,
        },
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    logger.info("saved %s", summary_path)

    # خلاصه نهایی
    print("\n" + "=" * 60)
    print("CALIBRATION COMPLETE")
    print("=" * 60)
    print(
        f"Default     MAPE: {default_validation['mape_weighted']:.2f}%   "
        f"POM: {default_validation['n_patterns_passed']}/5"
    )
    print(
        f"Calibrated  MAPE: {calibrated_validation['mape_weighted']:.2f}%   "
        f"POM: {calibrated_validation['n_patterns_passed']}/5"
    )
    if calibrated_validation["pass_threshold"]:
        print("✅ MAPE under 15% → calibration SUCCESSFUL")
    elif calibrated_validation["mape_weighted"] < 20.0:
        print("⏳ MAPE under 20% → calibration acceptable")
    else:
        print("❌ MAPE above 20% → consider expanding search space")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
