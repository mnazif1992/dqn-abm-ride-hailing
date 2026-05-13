"""
ماژول اعتبارسنجی کالیبراسیون ABM.

این ماژول دو معیار سنجش را پیاده‌سازی می‌کند:

1) POM (Pattern-Oriented Modelling) — معیار اصلی پذیرش طبق فصل ۳ بخش ۳-۵-۵
   و جدول ۳-۸: هر یک از ۵ الگو باید در بازه مجاز مطلق خود قرار گیرد.
   اگر همه ۵ الگو در بازه باشند، مدل از نظر POM «پذیرفته» می‌شود.

2) MAPE وزن‌دار — معیار کمکی برای رتبه‌بندی ترکیب‌ها.
   هنگامی که چندین ترکیب از POM عبور می‌کنند، آن با کمترین MAPE وزن‌دار
   به‌عنوان بهترین انتخاب می‌شود. وزن CR و DU برابر ۲ و سایر شاخص‌ها ۱.

طبق فصل ۳ جدول ۳-۸:
    | الگو          | هدف   | بازه مجاز مطلق |
    | CR            | 61.54%| ±3%            |
    | WT            | 2.51m | ±0.3 min       |
    | DU            | 63.64%| ±3%            |
    | cancel_rate   | 26%   | ±2%            |
    | no_driver_rate| 12.5% | ±2%            |

نکته: الگوی ششم (WT-باران سنگین +۳۹.۷٪) در این ماژول لحاظ نمی‌شود.
این الگو به‌عنوان روایی بیرونی پس از انتخاب config برنده در
یک تست جداگانه (دو-سناریویی Clear vs Heavy_Rain) ارزیابی می‌شود.

مرجع: فصل ۳، بخش ۳-۵-۵، جدول ۳-۸، Grimm et al. (2005).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# تعریف ۵ الگوی POM طبق جدول ۳-۸ فصل ۳
# ============================================================================

POM_PATTERNS: List[Dict[str, Any]] = [
    {
        "name": "CR",
        "abm_key": "CR",
        "target_key": "CR_baseline",
        "tol_abs": 0.03,
        "weight": 2.0,
        "label_fa": "نرخ تکمیل سفر",
    },
    {
        "name": "WT",
        "abm_key": "mean_WT_min",
        "target_key": "WT_baseline",
        "tol_abs": 0.30,
        "weight": 1.0,
        "label_fa": "میانگین زمان انتظار (دقیقه)",
    },
    {
        "name": "DU",
        "abm_key": "mean_DU",
        "target_key": "DU_baseline",
        "tol_abs": 0.03,
        "weight": 2.0,
        "label_fa": "نرخ بهره‌وری راننده",
    },
    {
        "name": "cancel_rate",
        "abm_key": "cancel_rate",
        "target_key": "cancel_rate",
        "tol_abs": 0.02,
        "weight": 1.0,
        "label_fa": "نرخ لغو مسافر",
    },
    {
        "name": "no_driver_rate",
        "abm_key": "no_driver_rate",
        "target_key": "no_driver_rate",
        "tol_abs": 0.02,
        "weight": 1.0,
        "label_fa": "نرخ عدم یافتن راننده",
    },
]


# ============================================================================
# توابع کمکی
# ============================================================================

def _safe_mape(observed: float, target: float, eps: float = 1e-9) -> float:
    """محاسبه MAPE درصدی با محافظت در برابر تقسیم بر صفر."""
    return abs(observed - target) / max(abs(target), eps) * 100.0


# ============================================================================
# تابع اصلی اعتبارسنجی
# ============================================================================

def validate_calibration(
    model_output: Dict[str, Any],
    targets: Dict[str, Any],
    mape_threshold: float = 15.0,
    return_details: bool = True,
) -> Dict[str, Any]:
    """
    اعتبارسنجی خروجی ABM با دو معیار POM + MAPE.

    ورودی:
        model_output: خروجی episode_summary() از RideHailingModel یا dict
            میانگین چند seed با کلیدهای: CR, mean_WT_min, mean_DU,
            cancel_rate, no_driver_rate
        targets: dict بارگذاری‌شده از targets.json
        mape_threshold: آستانه MAPE وزن‌دار (پیش‌فرض ۱۵٪)
        return_details: اگر True، جزئیات هر KPI برمی‌گرداند

    خروجی: dict با کلیدهای:
        pom_per_pattern, pom_pass, n_patterns_passed, n_patterns_total,
        mape_per_kpi, mape_weighted, mape_threshold, pass_threshold, details
    """
    pom_per_pattern: Dict[str, bool] = {}
    mape_per_kpi: Dict[str, float] = {}
    details: Dict[str, Dict[str, float]] = {}

    weighted_sum = 0.0
    weight_total = 0.0

    for pat in POM_PATTERNS:
        name = pat["name"]
        abm_key = pat["abm_key"]
        target_key = pat["target_key"]
        tol = pat["tol_abs"]
        weight = pat["weight"]

        if abm_key not in model_output:
            logger.warning(
                "KPI '%s' در خروجی ABM یافت نشد، مقدار صفر فرض می‌شود.",
                abm_key,
            )
            observed = 0.0
        else:
            observed = float(model_output[abm_key])

        if target_key not in targets:
            raise KeyError(f"کلید '{target_key}' در targets.json یافت نشد.")
        target = float(targets[target_key])

        abs_err = abs(observed - target)
        in_band = abs_err <= tol
        pom_per_pattern[name] = bool(in_band)

        mape = _safe_mape(observed, target)
        mape_per_kpi[name] = mape

        weighted_sum += weight * mape
        weight_total += weight

        details[name] = {
            "observed": observed,
            "target": target,
            "abs_error": abs_err,
            "tol_abs": tol,
            "mape_percent": mape,
            "in_band": float(in_band),
            "weight": weight,
        }

    mape_weighted = weighted_sum / weight_total if weight_total > 0 else float("inf")
    n_passed = sum(pom_per_pattern.values())
    pom_pass = n_passed == len(POM_PATTERNS)
    pass_threshold = mape_weighted < mape_threshold

    result: Dict[str, Any] = {
        "pom_per_pattern": pom_per_pattern,
        "pom_pass": bool(pom_pass),
        "n_patterns_passed": int(n_passed),
        "n_patterns_total": len(POM_PATTERNS),
        "mape_per_kpi": mape_per_kpi,
        "mape_weighted": float(mape_weighted),
        "mape_threshold": float(mape_threshold),
        "pass_threshold": bool(pass_threshold),
    }
    if return_details:
        result["details"] = details
    return result


# ============================================================================
# گزارش‌گیری
# ============================================================================

def format_validation_report(
    result: Dict[str, Any],
    header: Optional[str] = None,
) -> str:
    """قالب‌بندی نتیجه validate_calibration به‌صورت متن قابل خواندن."""
    lines: List[str] = []
    border = "─" * 72
    lines.append(border)
    if header:
        lines.append(header)
        lines.append(border)
    lines.append(
        f"{'KPI':<18}{'Observed':>12}{'Target':>12}{'|Err|':>10}"
        f"{'Tol':>10}{'MAPE%':>10}  POM"
    )
    lines.append(border)
    details = result.get("details", {})
    for pat in POM_PATTERNS:
        name = pat["name"]
        if name not in details:
            continue
        d = details[name]
        status = "✓" if d["in_band"] >= 1.0 else "✗"
        lines.append(
            f"{name:<18}{d['observed']:>12.4f}{d['target']:>12.4f}"
            f"{d['abs_error']:>10.4f}{d['tol_abs']:>10.4f}"
            f"{d['mape_percent']:>10.2f}    {status}"
        )
    lines.append(border)
    lines.append(
        f"POM patterns passed: {result['n_patterns_passed']}/"
        f"{result['n_patterns_total']}  →  "
        f"{'PASS' if result['pom_pass'] else 'FAIL'}"
    )
    lines.append(
        f"MAPE weighted: {result['mape_weighted']:.2f}%  "
        f"(threshold {result['mape_threshold']:.1f}%)  →  "
        f"{'PASS' if result['pass_threshold'] else 'FAIL'}"
    )
    lines.append(border)
    return "\n".join(lines)


# ============================================================================
# اجرای مستقل برای تست سریع
# ============================================================================

if __name__ == "__main__":
    import logging as _lg
    _lg.basicConfig(
        level=_lg.INFO,
        format="[%(levelname)s] %(name)s: %(message)s",
    )

    # تست ۱: خروجی نزدیک به هدف (باید POM pass باشد)
    fake_targets = {
        "CR_baseline": 0.6154,
        "WT_baseline": 2.5154,
        "DU_baseline": 0.6364,
        "cancel_rate": 0.2599,
        "no_driver_rate": 0.1247,
    }
    fake_output_good = {
        "CR": 0.62,
        "mean_WT_min": 2.4,
        "mean_DU": 0.65,
        "cancel_rate": 0.27,
        "no_driver_rate": 0.13,
    }
    res = validate_calibration(fake_output_good, fake_targets)
    print(format_validation_report(res, header="Test 1: close to target"))

    # تست ۲: خروجی فعلی ABM (باید fail شدید باشد)
    fake_output_current = {
        "CR": 0.9860,
        "mean_WT_min": 1.91,
        "mean_DU": 0.2576,
        "cancel_rate": 0.0104,
        "no_driver_rate": 0.0000,
    }
    res = validate_calibration(fake_output_current, fake_targets)
    print(format_validation_report(res, header="Test 2: current ABM (uncalibrated)"))
