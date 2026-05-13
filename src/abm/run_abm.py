"""
اسکریپت اجرای ABM — تست عملکرد یک شبیه‌سازی ۷۲۰ گامی.

نحوه اجرا (از ریشه پروژه):
    python -m src.abm.run_abm
    python -m src.abm.run_abm --seed 42 --steps 720
    python -m src.abm.run_abm --debug
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .model import RideHailingModel
from .utils import load_config, load_targets, load_zones, setup_logging

logger = logging.getLogger("run_abm")


def parse_args() -> argparse.Namespace:
    """تجزیه آرگومان‌های خط فرمان."""
    p = argparse.ArgumentParser(description="ABM ride-hailing simulation (Phase 5).")
    p.add_argument("--config", type=str,
                   default="experiments/configs/abm_default.yaml",
                   help="مسیر فایل config YAML")
    p.add_argument("--targets", type=str,
                   default="data/calibration/targets.json")
    p.add_argument("--zones", type=str,
                   default="data/calibration/zones.json")
    p.add_argument("--seed", type=int, default=None,
                   help="بذر تصادفی (override روی config)")
    p.add_argument("--steps", type=int, default=None,
                   help="تعداد گام (override روی config)")
    p.add_argument("--debug", action="store_true", help="فعال کردن DEBUG logging")
    p.add_argument("--save-csv", type=str, default=None,
                   help="مسیر ذخیره داده‌های جمع‌آوری شده (CSV)")
    return p.parse_args()


def run_simulation(config: dict, targets: dict, zones_data: dict,
                   log_every: int = 100) -> RideHailingModel:
    """
    اجرای یک شبیه‌سازی کامل.

    ورودی:
        config: dict پارامترها
        targets: dict اهداف کالیبراسیون
        zones_data: dict نواحی
        log_every: هر چند گام، گزارش پیشرفت چاپ شود

    خروجی:
        مدل بعد از پایان شبیه‌سازی (برای تحلیل بیشتر)
    """
    model = RideHailingModel(config=config, targets=targets, zones_data=zones_data)
    max_steps = config["max_steps"]

    logger.info("Running simulation: %d steps (%.0fh)",
                max_steps, max_steps * config["step_minutes"] / 60.0)

    for t in range(max_steps):
        model.step()
        if (t + 1) % log_every == 0 or t == 0:
            logger.info(
                "Step %d/%d — active passengers: %d, active drivers: %d, "
                "assigns: %d, completed cumul: %d",
                t + 1, max_steps,
                model._n_waiting(),
                model._n_available_drivers(),
                model.step_n_assignments,
                model.total_completed,
            )

    logger.info("Step %d/%d — simulation complete", max_steps, max_steps)
    return model


def print_summary(model: RideHailingModel) -> None:
    """چاپ خلاصه نهایی به فرمت استاندارد."""
    s = model.episode_summary()
    border = "═" * 47
    print(f"\n{border}")
    print(f"Episode Summary (seed={s['seed']})")
    print(border)
    print(f"  Total requests:       {s['total_requests']}")
    print(f"  Completed trips:      {s['total_completed']} "
          f"(CR = {s['CR']:.4f})")
    print(f"  Cancellations:        {s['total_cancelled']} "
          f"(cancel_rate = {s['cancel_rate']:.4f})")
    print(f"  No-driver rejections: {s['n_no_driver']} "
          f"(no_driver = {s['no_driver_rate']:.4f})")
    print(f"  Mean WT (min):        {s['mean_WT_min']:.4f}")
    print(f"  Mean DU:              {s['mean_DU']:.4f}")
    print(f"{border}\n")


def compare_with_targets(model: RideHailingModel, targets: dict) -> None:
    """مقایسه خروجی با ۶ معیار POM فصل ۳ (جدول ۳-۸)."""
    s = model.episode_summary()
    print("Comparison with POM calibration targets (Chapter 3, Table 3-8):")
    print(f"  CR:           {s['CR']:.4f} vs target {targets['CR_baseline']:.4f} "
          f"(error ≤ 3%)")
    print(f"  Mean WT:      {s['mean_WT_min']:.2f}min vs target "
          f"{targets['WT_baseline']:.2f}min (error ≤ 0.3min)")
    print(f"  Mean DU:      {s['mean_DU']:.4f} vs target "
          f"{targets['DU_baseline']:.4f} (error ≤ 3%)")
    print(f"  Cancel rate:  {s['cancel_rate']:.4f} vs target "
          f"{targets['cancel_rate']:.4f} (error ≤ 2%)")
    print(f"  No-driver:    {s['no_driver_rate']:.4f} vs target "
          f"{targets['no_driver_rate']:.4f} (error ≤ 2%)")
    print()


def main() -> int:
    args = parse_args()
    setup_logging(level=logging.DEBUG if args.debug else logging.INFO)

    # مسیرها نسبت به ریشه پروژه
    root = Path(__file__).resolve().parents[2]
    config_path = root / args.config
    targets_path = root / args.targets
    zones_path = root / args.zones

    logger.info("Loading config from %s", args.config)
    config = load_config(config_path)

    logger.info("Loading targets from %s", args.targets)
    targets = load_targets(targets_path)

    logger.info("Loading %d zones from %s",
                len(load_zones(zones_path).get("zones", [])), args.zones)
    zones_data = load_zones(zones_path)

    # override از آرگومان‌ها
    if args.seed is not None:
        config["seed"] = args.seed
    if args.steps is not None:
        config["max_steps"] = args.steps

    logger.info("Initializing RideHailingModel with seed=%d", config["seed"])

    model = run_simulation(config, targets, zones_data)
    print_summary(model)
    compare_with_targets(model, targets)

    # ذخیره اختیاری
    if args.save_csv:
        out_path = root / args.save_csv
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df = model.get_collected_data()
        df.to_csv(out_path, index=False)
        logger.info("Saved collected data to %s (rows=%d)", out_path, len(df))

    return 0


if __name__ == "__main__":
    sys.exit(main())
