"""
عامل مسافر (PassengerAgent) برای ABM تخصیص تاکسی آنلاین.

پیاده‌سازی طبق فصل ۳ بخش ۳-۵-۱-الف، با اصلاحات زیر بر مبنای داده‌ی واقعی:
- توزیع max_wait از Gamma(a=0.997, scale=8.087) (طبق targets.json) به‌جای Exp(1/4)
- قانون لغو در هر گام با p=0.7 وقتی wait_time > max_wait (Geometric در هر گام)

وضعیت‌ها و گذارها:
    waiting → assigned → in_trip → completed
    waiting → cancelled
    waiting → no_driver (اگر هیچ کاندیدی نباشد)
"""
from __future__ import annotations

import enum
import logging
from typing import TYPE_CHECKING, Optional

import numpy as np
from mesa import Agent

if TYPE_CHECKING:
    from .model import RideHailingModel

logger = logging.getLogger(__name__)


class PassengerStatus(str, enum.Enum):
    """وضعیت‌های ممکن یک مسافر."""
    WAITING = "waiting"
    ASSIGNED = "assigned"
    IN_TRIP = "in_trip"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    NO_DRIVER = "no_driver"   # هیچ کاندیدی پیدا نشد


class PassengerAgent(Agent):
    """
    عامل مسافر.

    ویژگی‌های ثابت در زمان تولد:
        unique_id: شناسه یکتا
        origin_lat, origin_lon: مختصات مبدأ
        dest_lat, dest_lon: مختصات مقصد
        origin_zone, dest_zone: شناسه ناحیه مبدأ و مقصد
        request_time: گام شبیه‌سازی هنگام ثبت درخواست
        max_wait: حداکثر زمان قابل تحمل (دقیقه) — Gamma(a=0.997, scale=8.087)
        price_sensitivity: حساسیت قیمتی [0, 1] — Uniform
        trip_distance_km: فاصله مبدأ تا مقصد

    ویژگی‌های پویا:
        status: وضعیت فعلی
        wait_time: زمان انتظار تجمعی (دقیقه)
        assigned_driver_id: شناسه راننده تخصیص‌یافته (در صورت وجود)
        reject_count: تعداد دفعاتی که توسط راننده‌ها رد شده
        complete_time: گام پایان سفر (یا None)
        cancel_time: گام لغو (یا None)
    """

    # احتمال لغو در هر گام پس از تخطی از max_wait
    P_CANCEL_PER_STEP: float = 0.7

    def __init__(
        self,
        unique_id: int,
        model: "RideHailingModel",
        origin_lat: float,
        origin_lon: float,
        dest_lat: float,
        dest_lon: float,
        origin_zone: int,
        dest_zone: int,
        request_time: int,
        max_wait: float,
        price_sensitivity: float,
        trip_distance_km: float,
    ) -> None:
        super().__init__(unique_id, model)
        self.origin_lat: float = float(origin_lat)
        self.origin_lon: float = float(origin_lon)
        self.dest_lat: float = float(dest_lat)
        self.dest_lon: float = float(dest_lon)
        self.origin_zone: int = int(origin_zone)
        self.dest_zone: int = int(dest_zone)
        self.request_time: int = int(request_time)
        self.max_wait: float = float(max_wait)
        self.price_sensitivity: float = float(price_sensitivity)
        self.trip_distance_km: float = float(trip_distance_km)

        self.status: PassengerStatus = PassengerStatus.WAITING
        self.wait_time: float = 0.0   # زمان انتظار در صف (queue) — دقیقه
        self.pickup_eta_min: float = 0.0  # زمان رسیدن راننده پس از تخصیص — دقیقه
        self.assigned_driver_id: Optional[int] = None
        self.reject_count: int = 0
        self.complete_time: Optional[int] = None
        self.cancel_time: Optional[int] = None

    @property
    def total_wait_time(self) -> float:
        """
        کل زمان انتظار = زمان در صف + زمان رسیدن راننده.

        این معیار با WT داده واقعی (waiting_time_min) قابل مقایسه است:
        از لحظه ثبت درخواست تا لحظه سوار شدن مسافر.
        """
        return self.wait_time + self.pickup_eta_min

    # ------------------------------------------------------------------
    # API برای model
    # ------------------------------------------------------------------

    def step(self) -> None:
        """
        گام مسافر در زمان‌بندی Mesa.

        فقط زمان انتظار افزایش می‌یابد. قانون لغو و تخصیص در model.step
        مدیریت می‌شود تا تخصیص ترتیبی (الگوریتم ۱ فصل ۳) ممکن باشد.
        """
        if self.status == PassengerStatus.WAITING:
            self.wait_time += self.model.step_minutes

    def check_cancellation(self) -> bool:
        """
        بررسی لغو در یک گام طبق قانون فصل ۳ + اصلاح.

        قانون: اگر waiting باشد و wait_time > max_wait، با احتمال 0.7
        مسافر لغو می‌کند. این بررسی در هر گام انجام می‌شود (Geometric).

        خروجی:
            True اگر لغو شد، False در غیر این صورت.
        """
        if self.status != PassengerStatus.WAITING:
            return False
        if self.wait_time <= self.max_wait:
            return False
        if self.model.rng.random() < self.P_CANCEL_PER_STEP:
            self.status = PassengerStatus.CANCELLED
            self.cancel_time = self.model.schedule.steps
            logger.debug("passenger %d cancelled at t=%d (wait=%.2f, max_wait=%.2f)",
                         self.unique_id, self.cancel_time, self.wait_time, self.max_wait)
            return True
        return False

    def mark_assigned(self, driver_id: int) -> None:
        """علامت‌گذاری مسافر به‌عنوان تخصیص‌یافته."""
        self.status = PassengerStatus.ASSIGNED
        self.assigned_driver_id = driver_id

    def mark_in_trip(self) -> None:
        """شروع سفر اصلی (پس از pickup)."""
        self.status = PassengerStatus.IN_TRIP

    def mark_completed(self) -> None:
        """پایان سفر."""
        self.status = PassengerStatus.COMPLETED
        self.complete_time = self.model.schedule.steps

    def mark_no_driver(self) -> None:
        """هیچ کاندیدی برای این مسافر یافت نشد."""
        self.status = PassengerStatus.NO_DRIVER

    def mark_rejected(self) -> None:
        """راننده پیشنهاد را رد کرد؛ مسافر به انتظار برمی‌گردد."""
        self.reject_count += 1
        # وضعیت waiting باقی می‌ماند

    def __repr__(self) -> str:
        return (f"<Passenger id={self.unique_id} status={self.status.value} "
                f"wait={self.wait_time:.1f}/{self.max_wait:.1f} zone={self.origin_zone}>")


# ============================================================================
# Sampler برای تولید مسافرها
# ============================================================================

def sample_passenger_max_wait(rng: np.random.Generator,
                              gamma_a: float = 0.997304960225585,
                              gamma_scale: float = 8.08709147663157) -> float:
    """
    نمونه‌گیری از توزیع Gamma برای patience مسافر (طبق targets.json).

    پارامترها از داده‌ی واقعی فروردین ۱۴۰۳ استخراج شده‌اند:
    میانگین ≈ 8.06 دقیقه، میانه ≈ 5.57 دقیقه.

    کف پایین برای جلوگیری از مقادیر بسیار کوچک: 0.5 دقیقه.
    """
    val = float(rng.gamma(shape=gamma_a, scale=gamma_scale))
    return max(val, 0.5)


def sample_price_sensitivity(rng: np.random.Generator) -> float:
    """نمونه‌گیری از Uniform(0, 1) برای حساسیت قیمتی."""
    return float(rng.uniform(0.0, 1.0))


# ============================================================================
# تست مستقل
# ============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from utils import setup_logging
    setup_logging()
    rng = np.random.default_rng(42)

    samples = np.array([sample_passenger_max_wait(rng) for _ in range(10_000)])
    logger.info("max_wait samples: mean=%.3f median=%.3f (target: 8.06, 5.57)",
                samples.mean(), np.median(samples))
    logger.info("  p25=%.2f p75=%.2f p95=%.2f",
                np.percentile(samples, 25),
                np.percentile(samples, 75),
                np.percentile(samples, 95))
