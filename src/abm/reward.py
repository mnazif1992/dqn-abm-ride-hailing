"""
محاسبه‌گر پاداش — مرحله ۷ (نسخه v2)، فصل ۳ §۳-۷-۳.

دو حالت پشتیبانی می‌شود:

1) mode='partial'  — رابطه ۳-۸ فصل ۳ (همان v1):
       r = −α·(pickup_eta / WT_baseline)
           + β·(1 / N_drivers_zone)
           − λ·𝟙[reject]

2) mode='shaped'   — نسخه‌ی v2 (پس از شکست v1: CR=0.41 < Greedy 0.61):
       اگر accepted:
           r = +accept_bonus
               − pickup_alpha·(pickup_eta / WT_baseline)
               + drivers_beta·(1 / N_drivers_zone)
       اگر rejected:
           r = −reject_lambda
       پاداش تکمیل (به‌صورت retroactive، گزینه D §۳-۷-۲):
           وقتی passenger.status == COMPLETED شود، +completion_bonus
           به همان گذارِ تخصیص در replay buffer افزوده می‌شود.

دلیل shaping: تابع v1 همگی-منفی بود و سیگنال مثبت برای «تکمیل سفر»
نداشت؛ عامل به optimum محلی ضعیف همگرا شد. shaping سیگنال صریح برای
پذیرش و تکمیل اضافه می‌کند (سازگار با §۳-۷-۳ که تفکیک پاداش جزئی/سراسری
را مجاز می‌داند — این یک iteration روش‌شناختی است نه تناقض).

همه‌ی پارامترها از بخش `reward` در abm_calibrated.yaml خوانده می‌شوند.
"""
from __future__ import annotations

from typing import Any, Dict


class RewardCalculator:
    """محاسبه‌ی پاداش فوری و پاداش تکمیل بر اساس mode پیکربندی."""

    def __init__(self, reward_cfg: Dict[str, Any], mode: str | None = None) -> None:
        cfg = dict(reward_cfg or {})
        self.mode: str = str(mode or cfg.get("mode", "shaped")).lower()
        if self.mode not in ("partial", "shaped"):
            raise ValueError(
                f"reward mode must be 'partial' or 'shaped', got '{self.mode}'"
            )

        # خط‌مبنا (پشتیبانی از هر دو نام‌گذاری wt_baseline / WT_baseline)
        self.wt_baseline: float = float(
            cfg.get("wt_baseline", cfg.get("WT_baseline", 2.5154))
        )

        # --- پارامترهای partial (رابطه ۳-۸) ---
        self.alpha: float = float(cfg.get("alpha", 0.6))
        self.beta: float = float(cfg.get("beta", 0.4))
        self.lambda_reject: float = float(cfg.get("lambda_reject", 0.2))

        # --- پارامترهای shaped (v2) ---
        self.accept_bonus: float = float(cfg.get("accept_bonus", 0.5))
        # v4 (sharper reward, May 17): ضریب جریمه pickup از 0.3 به 1.0 افزایش یافت.
        # تشخیص v2 نشان داد Q function flat می‌شود (margin<0.2) به علت سیگنال
        # ضعیف candidate-level. این تغییر تفاوت reward بین کاندیدها را از ~0.36
        # به ~1.20 افزایش می‌دهد (در reward کلی ~+2.5). این یک تست تشخیصی است
        # برای اثبات اینکه ضعف candidate-level signal علت همگرایی به flat Q بود.
        self.pickup_alpha: float = float(cfg.get("pickup_alpha", 1.0))
        self.drivers_beta: float = float(cfg.get("drivers_beta", 0.1))
        self.reject_lambda: float = float(cfg.get("reject_lambda", 0.3))
        self.completion_bonus_value: float = float(
            cfg.get("completion_bonus", 2.0)
        )

    # ------------------------------------------------------------------
    # پاداش فوری (هنگام تصمیم تخصیص)
    # ------------------------------------------------------------------

    def immediate(
        self,
        accepted: bool,
        pickup_eta_min: float,
        n_drivers_zone: int,
    ) -> float:
        """
        پاداش فوری برای یک تصمیم تخصیص.

        ورودی:
            accepted: آیا راننده پذیرفت
            pickup_eta_min: زمان تخمینی رسیدن راننده (دقیقه)؛ برای reject = ۰
            n_drivers_zone: تعداد راننده‌های available در ناحیه (≥۱)
        """
        n_zone = max(1, int(n_drivers_zone))
        wt = max(self.wt_baseline, 1e-6)

        if self.mode == "partial":
            # رابطه ۳-۸ فصل ۳
            return float(
                -self.alpha * (pickup_eta_min / wt)
                + self.beta * (1.0 / n_zone)
                - self.lambda_reject * (0.0 if accepted else 1.0)
            )

        # mode == 'shaped' (v2)
        if accepted:
            return float(
                self.accept_bonus
                - self.pickup_alpha * (pickup_eta_min / wt)
                + self.drivers_beta * (1.0 / n_zone)
            )
        return float(-self.reject_lambda)

    # ------------------------------------------------------------------
    # پاداش تکمیل (retroactive — گزینه D)
    # ------------------------------------------------------------------

    def completion_bonus(self) -> float:
        """
        مقدار پاداش تکمیل سفر.

        فقط در mode='shaped' فعال است؛ در mode='partial' صفر برمی‌گردد
        (رابطه ۳-۸ پاداش تکمیل ندارد).
        """
        if self.mode == "shaped":
            return float(self.completion_bonus_value)
        return 0.0
