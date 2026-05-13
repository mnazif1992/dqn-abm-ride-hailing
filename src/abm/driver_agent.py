"""
عامل راننده (DriverAgent) برای ABM تخصیص تاکسی آنلاین.

پیاده‌سازی طبق فصل ۳ بخش ۳-۵-۱-ب با مدل پذیرش گسترش‌یافته رابطه ۳-۵.

وضعیت‌ها و گذارها:
    offline → available (شروع شیفت)
    available → en_route (پس از پذیرش)
    en_route → busy (پس از pickup)
    busy → available (پس از تحویل مسافر)
    * → offline (پایان شیفت)
"""
from __future__ import annotations

import enum
import logging
from typing import TYPE_CHECKING, Optional

import numpy as np
from mesa import Agent

from .utils import haversine_km, normalize_surge, traffic_factor, weather_factors

if TYPE_CHECKING:
    from .model import RideHailingModel
    from .passenger_agent import PassengerAgent

logger = logging.getLogger(__name__)


class DriverStatus(str, enum.Enum):
    """وضعیت‌های ممکن یک راننده."""
    OFFLINE = "offline"
    AVAILABLE = "available"
    EN_ROUTE = "en_route"   # در حال رفتن به مبدأ مسافر
    BUSY = "busy"           # در حال حمل مسافر


class DriverAgent(Agent):
    """
    عامل راننده.

    ویژگی‌های ثابت:
        driver_id: شناسه (همان unique_id)
        acceptance_rate (AR_d): نرخ پایه پذیرش — Beta(8, 2) ≈ 0.8
        destination_aversion (δ_d): گریز از مقصدهای کم‌تقاضا — Uniform(0,1)
        short_trip_aversion (σ_d): گریز از سفرهای کوتاه — Uniform(0,1)
        zone_preference: ناحیه ترجیحی (برای موقعیت اولیه)
        shift_start, shift_end: ساعت شروع/پایان شیفت (0..23)

    ویژگی‌های پویا:
        lat, lon: موقعیت فعلی
        status: وضعیت
        current_passenger_id: مسافر فعلی (در en_route/busy)
        remaining_pickup_min: زمان باقی‌مانده تا pickup (دقیقه)
        remaining_trip_min: زمان باقی‌مانده تا تحویل (دقیقه)
        pickup_origin_lat/lon: مبدأ pickup برای درون‌یابی موقعیت
        trip_dest_lat/lon: مقصد سفر فعلی
        total_pickup_time: زمان کل en_route در این شیفت (برای DU)
        total_busy_time: زمان کل busy در این شیفت (برای DU)
        total_active_time: زمان آنلاین تا الان
        trips_completed: تعداد سفرهای تکمیل‌شده
    """

    D_MAX_KM: float = 5.0   # شعاع کاندیدی و پارامتر f_dist (فصل ۳)

    def __init__(
        self,
        unique_id: int,
        model: "RideHailingModel",
        lat: float,
        lon: float,
        acceptance_rate: float,
        destination_aversion: float,
        short_trip_aversion: float,
        zone_preference: int,
        shift_start: int,
        shift_end: int,
    ) -> None:
        super().__init__(unique_id, model)
        # ثابت‌ها
        self.driver_id: int = int(unique_id)
        self.acceptance_rate: float = float(acceptance_rate)
        self.destination_aversion: float = float(destination_aversion)
        self.short_trip_aversion: float = float(short_trip_aversion)
        self.zone_preference: int = int(zone_preference)
        self.shift_start: int = int(shift_start)
        self.shift_end: int = int(shift_end)

        # پویا
        self.lat: float = float(lat)
        self.lon: float = float(lon)
        self.status: DriverStatus = DriverStatus.OFFLINE
        self.current_passenger_id: Optional[int] = None
        self.remaining_pickup_min: float = 0.0
        self.remaining_trip_min: float = 0.0
        self.pickup_origin_lat: float = self.lat
        self.pickup_origin_lon: float = self.lon
        self.pickup_target_lat: float = self.lat
        self.pickup_target_lon: float = self.lon
        self.trip_origin_lat: float = self.lat
        self.trip_origin_lon: float = self.lon
        self.trip_dest_lat: float = self.lat
        self.trip_dest_lon: float = self.lon
        self.pickup_total_min: float = 0.0    # کل زمان pickup فعلی (برای interpolation)
        self.trip_total_min: float = 0.0      # کل زمان trip فعلی

        # متریک‌ها
        self.total_pickup_time: float = 0.0
        self.total_busy_time: float = 0.0
        self.total_active_time: float = 0.0
        self.trips_completed: int = 0

    # ------------------------------------------------------------------
    # شیفت
    # ------------------------------------------------------------------

    def is_on_shift(self, hour: int) -> bool:
        """آیا راننده در ساعت داده‌شده، در شیفت است؟ پشتیبانی از شیفت گذرنده از نیمه‌شب."""
        if self.shift_start <= self.shift_end:
            return self.shift_start <= hour < self.shift_end
        # شیفت گذرنده از نیمه‌شب: مثلاً 22 → 6
        return hour >= self.shift_start or hour < self.shift_end

    def update_shift_status(self, hour: int) -> None:
        """به‌روزرسانی online/offline بر اساس ساعت."""
        on_shift = self.is_on_shift(hour)
        if on_shift and self.status == DriverStatus.OFFLINE:
            self.status = DriverStatus.AVAILABLE
        elif not on_shift and self.status == DriverStatus.AVAILABLE:
            # فقط اگر بیکار است، می‌رود offline؛ اگر در حال سفر است، تمامش می‌کند
            self.status = DriverStatus.OFFLINE

    # ------------------------------------------------------------------
    # مدل پذیرش (رابطه ۳-۵)
    # ------------------------------------------------------------------

    def acceptance_probability(
        self,
        passenger: "PassengerAgent",
        pickup_dist_km: float,
        surge_multiplier: float,
        zone_attractiveness: float,
    ) -> float:
        """
        محاسبه احتمال پذیرش طبق رابطه ۳-۵:

            p_accept = AR_d × f_dist × f_surge × f_dest × f_length

        که:
            f_dist   = max(0, 1 - pickup_dist / 5)
            f_surge  = 0.5 + 0.5 × surge_normalized
            f_dest   = 1 - δ_d × (1 - zone_attractiveness(dest_zone))
            f_length = 1 - σ_d × 𝟙[trip_distance < 2 km]

        ورودی:
            passenger: عامل مسافر کاندید
            pickup_dist_km: فاصله راننده تا مسافر
            surge_multiplier: ضریب سرج فعلی (>=۱.۰)
            zone_attractiveness: جذابیت ناحیه مقصد در [0, 1]

        خروجی:
            p_accept در بازه [0, 1]
        """
        f_dist = max(0.0, 1.0 - pickup_dist_km / self.D_MAX_KM)
        f_surge = 0.5 + 0.5 * normalize_surge(surge_multiplier)
        f_dest = 1.0 - self.destination_aversion * (1.0 - zone_attractiveness)
        is_short = passenger.trip_distance_km < 2.0
        f_length = 1.0 - self.short_trip_aversion * (1.0 if is_short else 0.0)
        p = self.acceptance_rate * f_dist * f_surge * f_dest * f_length
        return float(np.clip(p, 0.0, 1.0))

    def decide_accept(self,
                      passenger: "PassengerAgent",
                      pickup_dist_km: float,
                      surge_multiplier: float,
                      zone_attractiveness: float) -> bool:
        """نمونه‌گیری برنولی از احتمال پذیرش."""
        p = self.acceptance_probability(passenger, pickup_dist_km,
                                        surge_multiplier, zone_attractiveness)
        accept = self.model.rng.random() < p
        logger.debug("driver %d -> passenger %d: p_accept=%.3f → %s",
                     self.driver_id, passenger.unique_id, p,
                     "ACCEPT" if accept else "REJECT")
        return accept

    # ------------------------------------------------------------------
    # تخصیص و حرکت
    # ------------------------------------------------------------------

    def assign_to(self,
                  passenger: "PassengerAgent",
                  base_speed_kmh: float,
                  hour: int,
                  weather: str) -> float:
        """
        تخصیص راننده به مسافر — وضعیت → en_route و محاسبه زمان pickup.

        زمان pickup = (فاصله / سرعت) × ضریب ترافیک × ضریب هوا

        خروجی: pickup_time به دقیقه (تا dispatcher روی passenger ست کند)
        """
        pickup_dist = haversine_km(self.lat, self.lon,
                                   passenger.origin_lat, passenger.origin_lon)
        trip_dist = passenger.trip_distance_km

        traffic = traffic_factor(hour)
        weather_traffic = weather_factors(weather)["traffic"]

        pickup_time = (pickup_dist / base_speed_kmh) * traffic * weather_traffic * 60.0
        trip_time = (trip_dist / base_speed_kmh) * traffic * weather_traffic * 60.0

        self.status = DriverStatus.EN_ROUTE
        self.current_passenger_id = passenger.unique_id
        self.remaining_pickup_min = pickup_time
        self.pickup_total_min = pickup_time
        self.pickup_origin_lat = self.lat
        self.pickup_origin_lon = self.lon
        self.pickup_target_lat = passenger.origin_lat
        self.pickup_target_lon = passenger.origin_lon
        self.trip_origin_lat = passenger.origin_lat
        self.trip_origin_lon = passenger.origin_lon
        self.trip_dest_lat = passenger.dest_lat
        self.trip_dest_lon = passenger.dest_lon
        self.remaining_trip_min = trip_time
        self.trip_total_min = trip_time

        logger.debug("driver %d assigned to passenger %d: pickup=%.1fmin trip=%.1fmin",
                     self.driver_id, passenger.unique_id, pickup_time, trip_time)
        return pickup_time

    def advance(self, dt_minutes: float) -> Optional[str]:
        """
        پیشروی یک گام زمانی برای راننده. موقعیت با درون‌یابی خطی به‌روز می‌شود.

        ورودی:
            dt_minutes: مدت گام (دقیقه)

        خروجی:
            رویداد ایجادشده:
              - "picked_up": راننده به مبدأ رسید (en_route → busy)
              - "completed": سفر تمام شد (busy → available)
              - None: تغییر فاز نداشتیم
        """
        event: Optional[str] = None

        if self.status == DriverStatus.AVAILABLE:
            self.total_active_time += dt_minutes
            return None

        if self.status == DriverStatus.EN_ROUTE:
            self.total_active_time += dt_minutes
            self.total_pickup_time += dt_minutes
            self.remaining_pickup_min -= dt_minutes
            if self.remaining_pickup_min <= 0:
                # رسیدیم به مبدأ مسافر
                self.lat = self.pickup_target_lat
                self.lon = self.pickup_target_lon
                self.status = DriverStatus.BUSY
                self.remaining_pickup_min = 0.0
                event = "picked_up"
            else:
                # درون‌یابی خطی روی مختصات (تقریب کافی برای شعاع چند کیلومتر)
                progress = 1.0 - (self.remaining_pickup_min / max(self.pickup_total_min, 1e-9))
                self.lat = (self.pickup_origin_lat
                            + progress * (self.pickup_target_lat - self.pickup_origin_lat))
                self.lon = (self.pickup_origin_lon
                            + progress * (self.pickup_target_lon - self.pickup_origin_lon))
            return event

        if self.status == DriverStatus.BUSY:
            self.total_active_time += dt_minutes
            self.total_busy_time += dt_minutes
            self.remaining_trip_min -= dt_minutes
            if self.remaining_trip_min <= 0:
                self.lat = self.trip_dest_lat
                self.lon = self.trip_dest_lon
                self.status = DriverStatus.AVAILABLE
                self.current_passenger_id = None
                self.remaining_trip_min = 0.0
                self.trips_completed += 1
                event = "completed"
            else:
                progress = 1.0 - (self.remaining_trip_min / max(self.trip_total_min, 1e-9))
                self.lat = (self.trip_origin_lat
                            + progress * (self.trip_dest_lat - self.trip_origin_lat))
                self.lon = (self.trip_origin_lon
                            + progress * (self.trip_dest_lon - self.trip_origin_lon))
            return event

        return None

    def step(self) -> None:
        """گام Mesa: بدنه‌ی اصلی در model.advance_drivers انجام می‌شود."""
        # خالی — تا تمام حرکات راننده‌ها به‌صورت متمرکز در model پردازش شوند
        pass

    def __repr__(self) -> str:
        return (f"<Driver id={self.driver_id} status={self.status.value} "
                f"loc=({self.lat:.4f},{self.lon:.4f}) AR={self.acceptance_rate:.2f}>")


# ============================================================================
# Samplers برای ایجاد رانندگان
# ============================================================================

def sample_acceptance_rate(rng: np.random.Generator) -> float:
    """نمونه‌گیری AR_d از Beta(8, 2) — میانگین ≈ 0.8 طبق فصل ۳."""
    return float(rng.beta(8.0, 2.0))


def sample_destination_aversion(rng: np.random.Generator) -> float:
    """δ_d ~ Uniform(0, 1) طبق فصل ۳."""
    return float(rng.uniform(0.0, 1.0))


def sample_short_trip_aversion(rng: np.random.Generator) -> float:
    """σ_d ~ Uniform(0, 1) طبق فصل ۳."""
    return float(rng.uniform(0.0, 1.0))


def sample_shift(rng: np.random.Generator,
                 mean_hours: float = 8.0,
                 std_hours: float = 2.0,
                 min_hours: int = 4,
                 max_hours: int = 12) -> tuple[int, int]:
    """
    تولید پنجره شیفت تصادفی.

    خروجی: (shift_start, shift_end) — ساعات صحیح 0..23.
    طول شیفت از Normal(mean_hours, std_hours) clipped به [min_hours, max_hours].
    """
    start = int(rng.integers(0, 24))
    length = int(np.clip(round(rng.normal(mean_hours, std_hours)), min_hours, max_hours))
    end = (start + length) % 24
    return start, end


# ============================================================================
# تست مستقل
# ============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from utils import setup_logging
    setup_logging()

    rng = np.random.default_rng(42)
    ar_samples = np.array([sample_acceptance_rate(rng) for _ in range(10_000)])
    logger.info("acceptance_rate: mean=%.3f std=%.3f (target Beta(8,2) ≈ 0.8±0.12)",
                ar_samples.mean(), ar_samples.std())

    # تست شیفت
    shifts = [sample_shift(rng) for _ in range(5)]
    for s, e in shifts:
        logger.info("shift: %d → %d", s, e)
