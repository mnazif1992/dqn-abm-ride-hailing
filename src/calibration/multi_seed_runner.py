"""
ماژول اجرای چندبار ABM با seedهای مختلف (Multi-seed Runner).

کارکرد:
1) override پارامترهای رفتاری ABM که در کد hard-code هستند (monkey-patching)
   بدون تغییر در هیچ‌یک از فایل‌های src/abm/.
2) اجرای ABM با چند seed و جمع‌آوری خروجی هر seed.
3) محاسبه میانگین و انحراف معیار KPIها برای کاهش واریانس.

پارامترهای قابل override:
    p_cancel_per_step     → PassengerAgent.P_CANCEL_PER_STEP
    patience_gamma_a      → default arg در sample_passenger_max_wait
    patience_gamma_scale  → default arg در sample_passenger_max_wait
    acceptance_beta_a     → ضریب a در Beta(a, b) در sample_acceptance_rate
    acceptance_beta_b     → ضریب b در Beta(a, b)
    demand_multiplier     → ضرب در requests_per_minute_avg
    n_drivers, d_max_km   → از طریق config (override طبیعی ABM)

نکته معماری: monkey-patching در یک context manager انجام می‌شود تا
پس از پایان اجرا، حالت سراسری بازگردانده شود. این برای اجرای موازی
(در grid_search) با multiprocessing امن است چون هر process حالت خود را دارد.

مرجع: درخواست کاربر در پیام اولیه — «بدون تغییر فایل‌های src/abm/».
"""
from __future__ import annotations

import contextlib
import logging
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Seedهای پیش‌فرض برای محاسبه میانگین و واریانس
# ============================================================================

DEFAULT_SEEDS: List[int] = [42, 123, 456, 789, 1010, 1337, 2024, 3141, 5926, 7777]


# ============================================================================
# Context manager برای override پارامترهای رفتاری ABM
# ============================================================================

@contextlib.contextmanager
def abm_param_overrides(
    p_cancel_per_step: Optional[float] = None,
    patience_gamma_a: Optional[float] = None,
    patience_gamma_scale: Optional[float] = None,
    acceptance_beta_a: Optional[float] = None,
    acceptance_beta_b: Optional[float] = None,
) -> Iterator[None]:
    """
    Context manager برای override پارامترهای hard-code شده در src/abm/.

    در ورود: مقادیر فعلی را ذخیره و مقادیر جدید را اعمال می‌کند.
    در خروج: مقادیر اصلی را بازمی‌گرداند تا حالت سراسری آلوده نشود.

    ورودی:
        p_cancel_per_step: احتمال لغو در هر گام (پیش‌فرض ABM: 0.7)
        patience_gamma_a: شکل توزیع Gamma برای patience (پیش‌فرض: 0.997)
        patience_gamma_scale: مقیاس Gamma (پیش‌فرض: 8.087)
        acceptance_beta_a: پارامتر a در Beta برای acceptance_rate (پیش‌فرض: 8)
        acceptance_beta_b: پارامتر b در Beta (پیش‌فرض: 2)

    مثال:
        with abm_param_overrides(p_cancel_per_step=0.5):
            model = RideHailingModel(config, targets, zones)
            # اجرا با p_cancel=0.5
        # خارج از بلوک، p_cancel به 0.7 بازگشته
    """
    from src.abm import passenger_agent, driver_agent

    # ---- ذخیره حالت اصلی ----
    saved: Dict[str, Any] = {}

    # 1) P_CANCEL_PER_STEP (class attribute)
    saved["P_CANCEL_PER_STEP"] = passenger_agent.PassengerAgent.P_CANCEL_PER_STEP

    # 2) sample_passenger_max_wait (default args)
    saved["sample_passenger_max_wait"] = passenger_agent.sample_passenger_max_wait

    # 3) sample_acceptance_rate (تابع کامل را جایگزین می‌کنیم چون پارامترها hard-code‌اند)
    saved["sample_acceptance_rate"] = driver_agent.sample_acceptance_rate

    # ---- اعمال override ها ----

    # 1) p_cancel_per_step
    if p_cancel_per_step is not None:
        passenger_agent.PassengerAgent.P_CANCEL_PER_STEP = float(p_cancel_per_step)

    # 2) patience distribution: یک wrapper جدید تعریف کن
    if patience_gamma_a is not None or patience_gamma_scale is not None:
        a_new = float(patience_gamma_a) if patience_gamma_a is not None else 0.997304960225585
        scale_new = float(patience_gamma_scale) if patience_gamma_scale is not None else 8.08709147663157

        def _patched_max_wait(
            rng: np.random.Generator,
            gamma_a: float = a_new,
            gamma_scale: float = scale_new,
        ) -> float:
            val = float(rng.gamma(shape=gamma_a, scale=gamma_scale))
            return max(val, 0.5)

        # نکته مهم: model.py مستقیماً sample_passenger_max_wait را import کرده،
        # پس باید در namespace آن ماژول هم patch شود.
        passenger_agent.sample_passenger_max_wait = _patched_max_wait
        try:
            from src.abm import model as _model_mod
            _model_mod.sample_passenger_max_wait = _patched_max_wait
        except ImportError:
            pass

    # 3) acceptance_rate Beta: تابع را جایگزین کن
    if acceptance_beta_a is not None or acceptance_beta_b is not None:
        a_b = float(acceptance_beta_a) if acceptance_beta_a is not None else 8.0
        b_b = float(acceptance_beta_b) if acceptance_beta_b is not None else 2.0

        def _patched_acceptance(
            rng: np.random.Generator,
            _a: float = a_b,
            _b: float = b_b,
        ) -> float:
            return float(rng.beta(_a, _b))

        driver_agent.sample_acceptance_rate = _patched_acceptance
        try:
            from src.abm import model as _model_mod
            _model_mod.sample_acceptance_rate = _patched_acceptance
        except ImportError:
            pass

    try:
        yield
    finally:
        # ---- بازگرداندن همه چیز ----
        passenger_agent.PassengerAgent.P_CANCEL_PER_STEP = saved["P_CANCEL_PER_STEP"]
        passenger_agent.sample_passenger_max_wait = saved["sample_passenger_max_wait"]
        driver_agent.sample_acceptance_rate = saved["sample_acceptance_rate"]
        # و در namespace model.py
        try:
            from src.abm import model as _model_mod
            _model_mod.sample_passenger_max_wait = saved["sample_passenger_max_wait"]
            _model_mod.sample_acceptance_rate = saved["sample_acceptance_rate"]
        except ImportError:
            pass


# ============================================================================
# اجرای یک‌بار ABM با پارامترهای دلخواه
# ============================================================================

def run_abm_once(
    base_config: Dict[str, Any],
    targets: Dict[str, Any],
    zones_data: Dict[str, Any],
    seed: int,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    اجرای یک شبیه‌سازی ABM با ترکیب پارامترهای دلخواه.

    ورودی:
        base_config: dict پایه از abm_default.yaml
        targets: dict اهداف کالیبراسیون (از targets.json)
        zones_data: dict نواحی (از zones.json)
        seed: بذر تصادفی این اجرا
        overrides: dict پارامترهای override که می‌تواند شامل:
            - n_drivers, d_max_km        → از طریق config
            - demand_multiplier           → ضرب در requests_per_minute_avg
            - p_cancel_per_step           → از طریق monkey-patching
            - patience_gamma_a, patience_gamma_scale
            - acceptance_beta_a, acceptance_beta_b
            - search_radius_km            → نام مستعار d_max_km
            - patience_scale              → نام مستعار patience_gamma_scale

    خروجی: dict خروجی episode_summary() + کلیدهای محاسبه‌شده acceptance_rate و seed
    """
    overrides = dict(overrides or {})

    # ---- تنظیم config ----
    config = dict(base_config)
    config["seed"] = int(seed)

    # نام‌های مستعار: مطابق دسته‌بندی درخواست کاربر
    if "search_radius_km" in overrides:
        overrides["d_max_km"] = overrides.pop("search_radius_km")
    if "patience_scale" in overrides:
        overrides["patience_gamma_scale"] = overrides.pop("patience_scale")

    # پارامترهایی که مستقیماً در config می‌روند:
    for key in (
        "n_drivers", "d_max_km", "step_minutes", "max_steps",
        "base_speed_kmh", "jitter_origin_deg", "jitter_driver_deg",
    ):
        if key in overrides:
            config[key] = overrides.pop(key)

    # demand_multiplier → روی targets اعمال می‌شود (ضرب در n_records معادل ضرب در rate)
    demand_mult = overrides.pop("demand_multiplier", None)
    targets_local = dict(targets)
    if demand_mult is not None and demand_mult != 1.0:
        # نسخه‌ی محلی targets با n_records ضرب‌شده — این تنها متغیری است که rate را تعیین می‌کند
        meta = dict(targets_local.get("metadata", {}))
        n_rec_old = float(meta.get("n_records", 104770))
        meta["n_records"] = n_rec_old * float(demand_mult)
        targets_local["metadata"] = meta

    # پارامترهای رفتاری برای monkey-patching
    patch_kwargs: Dict[str, Any] = {}
    for key in (
        "p_cancel_per_step", "patience_gamma_a", "patience_gamma_scale",
        "acceptance_beta_a", "acceptance_beta_b",
    ):
        if key in overrides:
            patch_kwargs[key] = overrides.pop(key)

    if overrides:
        logger.warning("override keys ignored (unknown): %s", list(overrides.keys()))

    # ---- اجرای ABM داخل context monkey-patching ----
    from src.abm.model import RideHailingModel

    with abm_param_overrides(**patch_kwargs):
        model = RideHailingModel(
            config=config,
            targets=targets_local,
            zones_data=zones_data,
        )
        max_steps = int(config.get("max_steps", 720))
        for _ in range(max_steps):
            model.step()
        summary = model.episode_summary()

        # محاسبه acceptance_rate از dispatcher
        n_assign = model.dispatcher.total_assignments
        n_reject = model.dispatcher.total_rejections
        denom = n_assign + n_reject
        summary["acceptance_rate"] = float(n_assign / denom) if denom > 0 else 0.0
        summary["total_assignments"] = int(n_assign)
        summary["total_rejections"] = int(n_reject)

        # تمیزکاری حافظه
        del model

    return summary


# ============================================================================
# اجرای چندبار و تجمیع
# ============================================================================

def run_multi_seed(
    base_config: Dict[str, Any],
    targets: Dict[str, Any],
    zones_data: Dict[str, Any],
    overrides: Optional[Dict[str, Any]] = None,
    seeds: Optional[List[int]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    اجرای ABM با چند seed و محاسبه میانگین/انحراف معیار KPIها.

    ورودی:
        base_config, targets, zones_data: مانند run_abm_once
        overrides: پارامترهای override
        seeds: لیست seedها (پیش‌فرض: 10 seed استاندارد)
        verbose: گزارش پیشرفت

    خروجی: dict با کلیدهای:
        - per_seed: List[dict] → خروجی هر seed
        - mean: dict → میانگین KPIها (سازگار با validate_calibration)
        - std: dict → انحراف معیار KPIها
        - n_seeds: تعداد seedهای استفاده‌شده
        - overrides: پارامترهای override برای ردیابی
    """
    seeds = list(seeds) if seeds is not None else DEFAULT_SEEDS

    per_seed: List[Dict[str, Any]] = []
    for i, s in enumerate(seeds):
        if verbose:
            logger.info("running seed %d/%d (seed=%d) ...", i + 1, len(seeds), s)
        out = run_abm_once(base_config, targets, zones_data, seed=s, overrides=overrides)
        per_seed.append(out)

    # ---- تجمیع: میانگین و std برای KPIهای عددی ----
    numeric_keys = [
        "CR", "cancel_rate", "no_driver_rate",
        "mean_WT_min", "mean_DU",
        "acceptance_rate",
        "total_requests", "total_completed", "total_cancelled",
        "n_no_driver", "total_assignments", "total_rejections",
    ]
    mean_out: Dict[str, float] = {}
    std_out: Dict[str, float] = {}
    for k in numeric_keys:
        vals = [float(d.get(k, 0.0)) for d in per_seed if k in d]
        if vals:
            mean_out[k] = float(np.mean(vals))
            std_out[k] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    return {
        "per_seed": per_seed,
        "mean": mean_out,
        "std": std_out,
        "n_seeds": len(seeds),
        "seeds": seeds,
        "overrides": overrides or {},
    }


# ============================================================================
# اجرای مستقل برای تست (نیاز به mesa)
# ============================================================================

if __name__ == "__main__":
    import json
    from pathlib import Path

    from src.abm.utils import load_config, load_targets, load_zones, setup_logging

    setup_logging()

    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "experiments" / "configs" / "abm_default.yaml")
    targets = load_targets(root / "data" / "calibration" / "targets.json")
    zones_data = load_zones(root / "data" / "calibration" / "zones.json")

    # تست ۱: یک‌بار اجرا با پارامترهای پیش‌فرض
    logger.info("=== test 1: single run with defaults ===")
    out = run_abm_once(config, targets, zones_data, seed=42)
    print(json.dumps(
        {k: v for k, v in out.items() if isinstance(v, (int, float))},
        indent=2,
    ))

    # تست ۲: ۳ seed با override
    logger.info("=== test 2: multi-seed with overrides ===")
    out = run_multi_seed(
        config, targets, zones_data,
        overrides={
            "demand_multiplier": 2.0,
            "n_drivers": 500,
            "p_cancel_per_step": 0.5,
        },
        seeds=[42, 123, 456],
    )
    print("mean:", json.dumps(out["mean"], indent=2))
    print("std:", json.dumps(out["std"], indent=2))
