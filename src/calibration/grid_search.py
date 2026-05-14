"""
اسکریپت Grid Search موازی برای کالیبراسیون ABM.

این اسکریپت تمام ترکیب‌های فضای جستجو را اجرا می‌کند، هر ترکیب را با
چند seed تکرار می‌کند، نتایج را با validate_calibration رتبه‌بندی می‌کند،
و بهترین config را در experiments/configs/abm_calibrated.yaml ذخیره می‌کند.

ویژگی‌ها:
    - موازی‌سازی با multiprocessing.Pool (تا تمام هسته‌های CPU)
    - Checkpoint دوره‌ای: اگر اجرا قطع شد، با فراخوانی مجدد ادامه می‌یابد
    - Progress bar با tqdm
    - Logging کامل
    - معیار رتبه‌بندی: ابتدا n_patterns_passed (بیشتر بهتر)، سپس mape_weighted (کمتر بهتر)

فضای جستجوی پیش‌فرض (تأیید کاربر در گفت‌و‌گو):
    - p_cancel_per_step:   [0.5, 0.7, 0.85]
    - patience_scale:      [4.0, 8.087, 12.0]
    - acceptance_beta_a:   [4.0, 6.0, 8.0]
    - demand_multiplier:   [1.0, 1.5, 2.0]
    - n_drivers:           [500, 650, 800]
    - search_radius_km:    [3.0, 5.0]
جمع: 3×3×3×3×3×2 = 486 ترکیب × 3 seed = 1458 اجرا (≈ ۶۰ دقیقه با ۸ هسته)

نحوه اجرا:
    python -m src.calibration.grid_search                # با تنظیمات پیش‌فرض
    python -m src.calibration.grid_search --n-workers 4  # محدود کردن worker
    python -m src.calibration.grid_search --no-resume    # شروع از صفر
"""
from __future__ import annotations

import argparse
import itertools
import logging
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ============================================================================
# فضای جستجوی پیش‌فرض (مطابق توافق با کاربر)
# ============================================================================

DEFAULT_SEARCH_SPACE: Dict[str, List[Any]] = {
    "p_cancel_per_step":   [0.5, 0.7, 0.85],
    "patience_scale":      [4.0, 8.087, 12.0],
    "acceptance_beta_a":   [4.0, 6.0, 8.0],
    "demand_multiplier":   [1.0, 1.5, 2.0],
    "n_drivers":           [500, 650, 800],
    "search_radius_km":    [3.0, 5.0],
}

DEFAULT_GRID_SEEDS: List[int] = [42, 123, 456]   # ۳ seed در grid search


# ============================================================================
# تولید ترکیب‌ها
# ============================================================================

def build_combinations(search_space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    تولید تمام ترکیب‌های متقاطع (Cartesian product) فضای جستجو.

    خروجی: لیست dict — هر dict یک ترکیب پارامتر.
    """
    keys = list(search_space.keys())
    value_lists = [search_space[k] for k in keys]
    combos: List[Dict[str, Any]] = []
    for vals in itertools.product(*value_lists):
        combos.append({k: v for k, v in zip(keys, vals)})
    return combos


def combo_to_key(combo: Dict[str, Any]) -> str:
    """تبدیل یک ترکیب به کلید قابل خواندن (برای checkpoint و logging)."""
    parts = [f"{k}={v}" for k, v in sorted(combo.items())]
    return "|".join(parts)


# ============================================================================
# اجرای یک ترکیب (worker function)
# ============================================================================

def _evaluate_combo(
    args: Tuple[int, Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], List[int]]
) -> Dict[str, Any]:
    """
    تابع worker برای multiprocessing — یک ترکیب را با چند seed اجرا می‌کند.

    این تابع باید در سطح ماژول باشد (نه nested) چون multiprocessing
    آن را با pickle منتقل می‌کند.

    ورودی args: tuple شامل
        - combo_idx: شناسه ترکیب
        - combo: dict پارامترها
        - base_config: config پایه
        - targets: targets.json
        - zones_data: zones.json
        - seeds: لیست seedها

    خروجی: dict نتیجه شامل combo، آمار، MAPE و POM
    """
    combo_idx, combo, base_config, targets, zones_data, seeds = args
    worker_logger = logging.getLogger(f"worker.{os.getpid()}")
    worker_logger.setLevel(logging.WARNING)   # کاهش نویز در child processها

    # import در داخل worker تا multiprocessing 'spawn' هم کار کند
    from src.calibration.multi_seed_runner import run_multi_seed
    from src.calibration.validate import validate_calibration

    t0 = time.time()
    try:
        multi = run_multi_seed(
            base_config=base_config,
            targets=targets,
            zones_data=zones_data,
            overrides=combo,
            seeds=seeds,
            verbose=False,
        )
        mean_out = multi["mean"]
        std_out = multi["std"]
        val = validate_calibration(mean_out, targets, return_details=False)

        result = {
            "combo_idx": combo_idx,
            "combo_key": combo_to_key(combo),
            # ترکیب پارامتر
            **{f"param_{k}": v for k, v in combo.items()},
            # KPIهای ABM
            "CR_mean": mean_out.get("CR"),
            "CR_std": std_out.get("CR"),
            "WT_mean": mean_out.get("mean_WT_min"),
            "WT_std": std_out.get("mean_WT_min"),
            "DU_mean": mean_out.get("mean_DU"),
            "DU_std": std_out.get("mean_DU"),
            "cancel_rate_mean": mean_out.get("cancel_rate"),
            "no_driver_rate_mean": mean_out.get("no_driver_rate"),
            "acceptance_rate_mean": mean_out.get("acceptance_rate"),
            # متریک‌های POM/MAPE
            "n_patterns_passed": val["n_patterns_passed"],
            "pom_pass": val["pom_pass"],
            "mape_weighted": val["mape_weighted"],
            "pass_threshold": val["pass_threshold"],
            "mape_CR": val["mape_per_kpi"].get("CR"),
            "mape_WT": val["mape_per_kpi"].get("WT"),
            "mape_DU": val["mape_per_kpi"].get("DU"),
            "mape_cancel_rate": val["mape_per_kpi"].get("cancel_rate"),
            "mape_no_driver_rate": val["mape_per_kpi"].get("no_driver_rate"),
            # متادادی
            "n_seeds": multi["n_seeds"],
            "elapsed_s": round(time.time() - t0, 2),
            "error": None,
        }
        return result
    except Exception as e:
        worker_logger.exception("combo %d failed: %s", combo_idx, e)
        return {
            "combo_idx": combo_idx,
            "combo_key": combo_to_key(combo),
            **{f"param_{k}": v for k, v in combo.items()},
            "error": str(e),
            "elapsed_s": round(time.time() - t0, 2),
        }


# ============================================================================
# Checkpoint
# ============================================================================

def _load_checkpoint(path: Path) -> Tuple[List[Dict[str, Any]], set]:
    """خواندن نتایج قبلی از CSV checkpoint (در صورت وجود)."""
    if not path.exists():
        return [], set()
    try:
        df = pd.read_csv(path)
        records = df.to_dict(orient="records")
        done_keys = set(df["combo_key"].astype(str).tolist())
        logger.info(
            "loaded checkpoint: %d completed combos from %s",
            len(records), path,
        )
        return records, done_keys
    except Exception as e:
        logger.warning("could not read checkpoint %s: %s — starting fresh", path, e)
        return [], set()


def _save_results(records: List[Dict[str, Any]], path: Path) -> None:
    """ذخیره نتایج تجمعی به CSV (atomic via tmp)."""
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


# ============================================================================
# انتخاب بهترین config
# ============================================================================

def select_best(df: pd.DataFrame) -> Dict[str, Any]:
    """
    انتخاب بهترین ترکیب از DataFrame نتایج.

    رتبه‌بندی:
        ۱) بیشینه n_patterns_passed (هم‌خوانی با POM فصل ۳)
        ۲) سپس کمینه mape_weighted (تای‌بریکر)

    خروجی: dict ترکیب پارامتر بهترین config + متریک‌های آن.
    """
    valid = df[df["error"].isna()] if "error" in df.columns else df
    if valid.empty:
        raise RuntimeError("No valid results to select from.")
    sorted_df = valid.sort_values(
        by=["n_patterns_passed", "mape_weighted"],
        ascending=[False, True],
    )
    return sorted_df.iloc[0].to_dict()


def save_calibrated_config(
    best_row: Dict[str, Any],
    base_config: Dict[str, Any],
    out_path: Path,
) -> Dict[str, Any]:
    """
    ذخیره بهترین config در YAML برای استفاده در مراحل بعدی.

    خروجی: dict پیکربندی کالیبره‌شده (base + override بهترین ترکیب).
    """
    calibrated = dict(base_config)
    # کلیدهای param_* را به config اضافه/جایگزین می‌کنیم
    overrides_applied: Dict[str, Any] = {}
    for k, v in best_row.items():
        if not k.startswith("param_"):
            continue
        param_name = k[len("param_"):]
        overrides_applied[param_name] = v
        # نام‌های مستعار → کلیدهای config
        if param_name == "search_radius_km":
            calibrated["d_max_km"] = float(v)
        elif param_name == "n_drivers":
            calibrated["n_drivers"] = int(v)
        # سایر پارامترها (p_cancel_per_step و ...) به‌عنوان «calibration» در config

    calibrated["calibration"] = {
        "applied_at": pd.Timestamp.now().isoformat(),
        "overrides": overrides_applied,
        "metrics": {
            "n_patterns_passed": int(best_row.get("n_patterns_passed", 0)),
            "pom_pass": bool(best_row.get("pom_pass", False)),
            "mape_weighted": float(best_row.get("mape_weighted", float("nan"))),
            "CR_mean": float(best_row.get("CR_mean", float("nan"))),
            "WT_mean": float(best_row.get("WT_mean", float("nan"))),
            "DU_mean": float(best_row.get("DU_mean", float("nan"))),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(calibrated, f, allow_unicode=True, sort_keys=False)
    logger.info("saved calibrated config to %s", out_path)
    return calibrated


# ============================================================================
# Main grid search
# ============================================================================

def run_grid_search(
    base_config: Dict[str, Any],
    targets: Dict[str, Any],
    zones_data: Dict[str, Any],
    search_space: Optional[Dict[str, List[Any]]] = None,
    seeds_per_combo: Optional[List[int]] = None,
    n_workers: Optional[int] = None,
    results_csv: Optional[Path] = None,
    calibrated_yaml: Optional[Path] = None,
    save_every: int = 10,
    resume: bool = True,
) -> pd.DataFrame:
    """
    اجرای Grid Search کامل.

    ورودی:
        base_config: config پایه (از abm_default.yaml)
        targets: targets.json
        zones_data: zones.json
        search_space: فضای جستجو (پیش‌فرض: DEFAULT_SEARCH_SPACE)
        seeds_per_combo: seedهای هر ترکیب (پیش‌فرض: [42, 123, 456])
        n_workers: تعداد processها (پیش‌فرض: cpu_count())
        results_csv: مسیر ذخیره نتایج تجمعی
        calibrated_yaml: مسیر ذخیره config برنده
        save_every: ذخیره دوره‌ای هر چند ترکیب
        resume: اگر True، نتایج موجود را بارگذاری کرده و فقط ترکیب‌های نمانده اجرا می‌شوند

    خروجی: DataFrame نتایج کامل (شامل ترکیب‌های قبلی در صورت resume).
    """
    space = search_space or DEFAULT_SEARCH_SPACE
    seeds = list(seeds_per_combo or DEFAULT_GRID_SEEDS)
    n_workers = int(n_workers or mp.cpu_count())
    results_csv = Path(results_csv) if results_csv else Path("experiments/logs/grid_search_results.csv")
    calibrated_yaml = Path(calibrated_yaml) if calibrated_yaml else Path("experiments/configs/abm_calibrated.yaml")

    combos = build_combinations(space)
    logger.info(
        "Grid search: %d combos × %d seeds = %d simulations",
        len(combos), len(seeds), len(combos) * len(seeds),
    )
    logger.info("Workers: %d", n_workers)

    # ---- resume از checkpoint ----
    existing_records: List[Dict[str, Any]] = []
    done_keys: set = set()
    if resume:
        existing_records, done_keys = _load_checkpoint(results_csv)
    pending = [(i, c) for i, c in enumerate(combos) if combo_to_key(c) not in done_keys]
    if existing_records:
        logger.info("Resuming: %d done, %d remaining", len(done_keys), len(pending))

    if not pending:
        logger.info("All combos already done. Loading and selecting best.")
        df = pd.DataFrame(existing_records)
    else:
        # ---- بسته‌بندی args برای worker ----
        args_iter = [
            (idx, combo, base_config, targets, zones_data, seeds)
            for idx, combo in pending
        ]

        records: List[Dict[str, Any]] = list(existing_records)

        # ---- اجرای موازی با progress bar ----
        ctx = mp.get_context("spawn")   # spawn ایمن‌تر برای کد علمی
        with ctx.Pool(processes=n_workers) as pool:
            try:
                with tqdm(total=len(args_iter), desc="Grid search", unit="combo") as pbar:
                    for i, result in enumerate(pool.imap_unordered(_evaluate_combo, args_iter)):
                        records.append(result)
                        pbar.update(1)
                        # ذخیره دوره‌ای
                        if (i + 1) % save_every == 0:
                            _save_results(records, results_csv)
            except KeyboardInterrupt:
                logger.warning("Interrupted by user — saving partial results.")
                pool.terminate()
                pool.join()
                _save_results(records, results_csv)
                raise

        # ذخیره نهایی
        _save_results(records, results_csv)
        df = pd.DataFrame(records)

    # ---- انتخاب و ذخیره بهترین ----
    if not df.empty:
        best = select_best(df)
        logger.info(
            "Best combo: pom_passed=%d/5  mape_weighted=%.2f%%",
            int(best.get("n_patterns_passed", 0)),
            float(best.get("mape_weighted", float("nan"))),
        )
        save_calibrated_config(best, base_config, calibrated_yaml)

    return df


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    """تجزیه آرگومان‌های خط فرمان."""
    p = argparse.ArgumentParser(description="Grid Search calibration for ABM.")
    p.add_argument("--config", type=str,
                   default="experiments/configs/abm_default.yaml")
    p.add_argument("--targets", type=str,
                   default="data/calibration/targets.json")
    p.add_argument("--zones", type=str, default="data/calibration/zones.json")
    p.add_argument("--results-csv", type=str,
                   default="experiments/logs/grid_search_results.csv")
    p.add_argument("--calibrated-yaml", type=str,
                   default="experiments/configs/abm_calibrated.yaml")
    p.add_argument("--n-workers", type=int, default=None,
                   help="تعداد worker (پیش‌فرض: cpu_count)")
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--no-resume", action="store_true",
                   help="نادیده گرفتن checkpoint و شروع از صفر")
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

    df = run_grid_search(
        base_config=config,
        targets=targets,
        zones_data=zones_data,
        n_workers=args.n_workers,
        results_csv=root / args.results_csv,
        calibrated_yaml=root / args.calibrated_yaml,
        save_every=args.save_every,
        resume=not args.no_resume,
    )

    if df.empty:
        logger.error("Grid search produced no results.")
        return 1

    # خلاصه نهایی
    valid = df[df["error"].isna()] if "error" in df.columns else df
    n_pom_pass = int((valid["pom_pass"] == True).sum()) if "pom_pass" in valid else 0
    n_mape_pass = int((valid["pass_threshold"] == True).sum()) if "pass_threshold" in valid else 0
    logger.info(
        "Summary: %d/%d combos pass POM,  %d/%d combos pass MAPE<15%%",
        n_pom_pass, len(valid), n_mape_pass, len(valid),
    )
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
