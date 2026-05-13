"""
مدل اصلی شبیه‌سازی تاکسی آنلاین تهران (RideHailingModel).

پیاده‌سازی طبق فصل ۳ بخش ۳-۵، با تخصیص ترتیبی (الگوریتم ۱).

اجزای اصلی:
    - ۱۴ ناحیه تهران (از zones.json)
    - فرآیند پواسون ناهمگن λ(zone, hour, weather)
    - مدل ترافیک ساعت‌محور
    - مدل هوای ۴ کلاسه (clear, cloudy, rainy, heavy_rain)
    - عوامل: PassengerAgent, DriverAgent, DispatcherAgent
    - DataCollector برای WT, DU, CR

گام زمانی: ۲ دقیقه، افق: ۷۲۰ گام (۲۴ ساعت).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation

from .dispatcher_agent import AssignmentStrategy, DispatcherAgent, GreedyStrategy
from .driver_agent import (
    DriverAgent,
    DriverStatus,
    sample_acceptance_rate,
    sample_destination_aversion,
    sample_shift,
    sample_short_trip_aversion,
)
from .passenger_agent import (
    PassengerAgent,
    PassengerStatus,
    sample_passenger_max_wait,
    sample_price_sensitivity,
)
from .utils import (
    haversine_km,
    haversine_matrix_km,
    map_weather,
    weather_factors,
    zones_to_arrays,
)

logger = logging.getLogger(__name__)


# ============================================================================
# RideHailingModel
# ============================================================================

class RideHailingModel(Model):
    """
    مدل اصلی ABM طبق فصل ۳ بخش ۳-۵.

    پارامترهای کلیدی (از config):
        n_drivers: تعداد کل راننده‌ها (پیش‌فرض ۸۰۰)
        step_minutes: مدت هر گام (۲ دقیقه)
        max_steps: افق شبیه‌سازی (۷۲۰ گام)
        base_speed_kmh: سرعت پایه (۳۰ km/h)
        d_max_km: شعاع کاندیدی (۵ km)
        seed: بذر تصادفی

    داده‌های ورودی:
        targets: dict از targets.json
        zones: dict از zones.json
    """

    def __init__(
        self,
        config: Dict[str, Any],
        targets: Dict[str, Any],
        zones_data: Dict[str, Any],
        strategy: Optional[AssignmentStrategy] = None,
    ) -> None:
        super().__init__()

        # -------- پارامترها --------
        self.config = config
        self.targets = targets
        self.zones_data = zones_data

        self.n_drivers: int = int(config.get("n_drivers", 800))
        self.step_minutes: float = float(config.get("step_minutes", 2.0))
        self.max_steps: int = int(config.get("max_steps", 720))
        self.base_speed_kmh: float = float(config.get("base_speed_kmh", 30.0))
        self.d_max_km: float = float(config.get("d_max_km", 5.0))
        self.seed_val: int = int(config.get("seed", 42))
        self.jitter_origin_deg: float = float(config.get("jitter_origin_deg", 0.005))
        self.jitter_driver_deg: float = float(config.get("jitter_driver_deg", 0.003))
        self.trip_distance_mean_km: float = float(targets.get("trip_distance_mean", 8.8))

        # -------- RNG مرکزی --------
        self.rng: np.random.Generator = np.random.default_rng(self.seed_val)
        # mesa.Model.random هم به همین seed تنظیم می‌شود
        self.random.seed(self.seed_val)

        # -------- نواحی --------
        zone_lats, zone_lons, zone_share = zones_to_arrays(zones_data)
        self.zone_lats: np.ndarray = zone_lats
        self.zone_lons: np.ndarray = zone_lons
        self.zone_share: np.ndarray = zone_share
        self.n_zones: int = len(zone_lats)

        # ماتریس فاصله بین نواحی (n×n کیلومتر)
        self.zone_dist_matrix: np.ndarray = haversine_matrix_km(zone_lats, zone_lons)

        # -------- تقاضای ناحیه × ساعت --------
        hourly = targets.get("hourly_demand_factor", {})
        self.hourly_factor: np.ndarray = np.array(
            [hourly.get(str(h), hourly.get(h, 1.0)) for h in range(24)],
            dtype=np.float64,
        )

        # -------- توزیع هوا --------
        weather_dist_raw: Dict[str, float] = targets.get("weather_distribution", {})
        self._weather_labels_raw: List[str] = list(weather_dist_raw.keys())
        self._weather_probs: np.ndarray = np.array(
            list(weather_dist_raw.values()), dtype=np.float64
        )
        self._weather_probs /= self._weather_probs.sum()
        # شرایط هوای فعلی (در هر گام به‌روز می‌شود)
        self.current_weather: str = "clear"

        # نرخ کل درخواست بر دقیقه از داده‌ی واقعی
        # n_records / (30 days × 24h × 60min) ≈ 2.42 درخواست/دقیقه
        n_records = targets.get("metadata", {}).get("n_records", 104770)
        self.requests_per_minute_avg: float = n_records / (30.0 * 24.0 * 60.0)

        # surge ثابت ≈ 1.0 از داده (هیچ نوسانی در داده نیست)
        self.current_surge: float = 1.0

        # -------- سرج/جذابیت ناحیه (cache) --------
        self._zone_attractiveness: np.ndarray = self._compute_zone_attractiveness(hour=0)

        # -------- زمان‌بند Mesa --------
        self.schedule = RandomActivation(self)

        # -------- ساخت عوامل --------
        self._next_id: int = 0
        self.passengers: List[PassengerAgent] = []
        self.drivers: List[DriverAgent] = []

        self._init_drivers()

        # Dispatcher
        if strategy is None:
            strategy = GreedyStrategy()
        self.dispatcher: DispatcherAgent = DispatcherAgent(
            unique_id=self._gen_id(),
            model=self,
            strategy=strategy,
            d_max_km=self.d_max_km,
        )
        self.schedule.add(self.dispatcher)

        # -------- متریک‌های تجمعی --------
        self.total_requests: int = 0
        self.total_completed: int = 0
        self.total_cancelled: int = 0
        self.total_no_driver: int = 0

        # متریک‌های گام (برای DataCollector)
        self.step_wt_minutes: float = 0.0    # میانگین WT جفت‌های تشکیل‌شده در گام
        self.step_du: float = 0.0
        self.step_cr: float = 0.0
        self.step_n_completed: int = 0
        self.step_n_cancelled: int = 0
        self.step_n_new_requests: int = 0
        self.step_n_assignments: int = 0

        # -------- DataCollector --------
        self.datacollector = DataCollector(
            model_reporters={
                "step": lambda m: m.schedule.steps,
                "hour": lambda m: m.current_hour(),
                "weather": lambda m: m.current_weather,
                "n_active_passengers": lambda m: m._n_waiting(),
                "n_active_drivers": lambda m: m._n_available_drivers(),
                "n_assignments_step": lambda m: m.step_n_assignments,
                "n_new_requests": lambda m: m.step_n_new_requests,
                "n_completed_step": lambda m: m.step_n_completed,
                "n_cancelled_step": lambda m: m.step_n_cancelled,
                "WT_step": lambda m: m.step_wt_minutes,
                "DU": lambda m: m.step_du,
                "CR": lambda m: m.step_cr,
            }
        )

        logger.info(
            "RideHailingModel initialized: n_zones=%d, n_drivers=%d, "
            "step=%dmin, horizon=%d, seed=%d",
            self.n_zones, self.n_drivers, int(self.step_minutes),
            self.max_steps, self.seed_val,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _gen_id(self) -> int:
        nid = self._next_id
        self._next_id += 1
        return nid

    def current_hour(self) -> int:
        """ساعت روز بر اساس گام جاری."""
        minutes_passed = self.schedule.steps * self.step_minutes
        return int((minutes_passed // 60) % 24)

    def _n_waiting(self) -> int:
        return sum(1 for p in self.passengers if p.status == PassengerStatus.WAITING)

    def _n_available_drivers(self) -> int:
        return sum(1 for d in self.drivers if d.status == DriverStatus.AVAILABLE)

    # ------------------------------------------------------------------
    # ساخت رانندگان
    # ------------------------------------------------------------------

    def _init_drivers(self) -> None:
        """ایجاد ۸۰۰ راننده با ویژگی‌های توزیع‌شده."""
        # تخصیص ناحیه ترجیحی بر اساس zone_demand_share
        share = self.zone_share / self.zone_share.sum()
        for _ in range(self.n_drivers):
            zone_pref = int(self.rng.choice(self.n_zones, p=share))
            # موقعیت اولیه: مرکز ناحیه + jitter گاوسی
            lat = self.zone_lats[zone_pref] + self.rng.normal(0, self.jitter_driver_deg)
            lon = self.zone_lons[zone_pref] + self.rng.normal(0, self.jitter_driver_deg)
            shift_start, shift_end = sample_shift(self.rng)

            driver = DriverAgent(
                unique_id=self._gen_id(),
                model=self,
                lat=lat,
                lon=lon,
                acceptance_rate=sample_acceptance_rate(self.rng),
                destination_aversion=sample_destination_aversion(self.rng),
                short_trip_aversion=sample_short_trip_aversion(self.rng),
                zone_preference=zone_pref,
                shift_start=shift_start,
                shift_end=shift_end,
            )
            self.drivers.append(driver)
            self.schedule.add(driver)

        # وضعیت اولیه شیفت در ساعت 0
        for d in self.drivers:
            d.update_shift_status(hour=0)

        logger.info("created %d drivers (n_online at hour 0: %d)",
                    len(self.drivers), self._n_available_drivers())

    # ------------------------------------------------------------------
    # جذابیت ناحیه
    # ------------------------------------------------------------------

    def _compute_zone_attractiveness(self, hour: int) -> np.ndarray:
        """
        جذابیت هر ناحیه = نرمال‌شده‌ی (zone_share × hourly_factor[next_hour]).

        طبق فصل ۳: «تقاضای پیش‌بینی‌شده ناحیه z در ۳۰ دقیقه آینده».
        با گام ۲ دقیقه، تقریباً ساعت بعد.
        """
        next_hour = (hour + 1) % 24
        raw = self.zone_share * self.hourly_factor[next_hour]
        # نرمال‌سازی به [0, 1] با Min-Max
        rmin, rmax = float(raw.min()), float(raw.max())
        if rmax - rmin < 1e-9:
            return np.full_like(raw, 0.5)
        return (raw - rmin) / (rmax - rmin)

    # ------------------------------------------------------------------
    # تولید درخواست (پواسون ناهمگن)
    # ------------------------------------------------------------------

    def _generate_requests(self, hour: int) -> int:
        """
        تولید مسافران جدید در این گام بر اساس فرآیند پواسون ناهمگن.

        نرخ کل = requests_per_minute × step_minutes × hourly_factor × weather_demand
        سپس به نواحی با احتمال zone_share توزیع می‌شود.
        """
        wf = weather_factors(self.current_weather)
        rate_total = (
            self.requests_per_minute_avg
            * self.step_minutes
            * self.hourly_factor[hour]
            * wf["demand"]
        )
        n_new = int(self.rng.poisson(rate_total))
        if n_new == 0:
            return 0

        # تخصیص نواحی مبدأ
        share = self.zone_share / self.zone_share.sum()
        origin_zones = self.rng.choice(self.n_zones, size=n_new, p=share)
        # مقاصد: همان توزیع share (تقریب رتبه ۱، در صورت نیاز با OD matrix جایگزین شود)
        dest_zones = self.rng.choice(self.n_zones, size=n_new, p=share)

        for o_zone, d_zone in zip(origin_zones, dest_zones):
            # جلوگیری از O=D با احتمال بالا (اگر اتفاق افتاد، مجدد سمپل کن)
            if o_zone == d_zone:
                # یک تلاش دیگر
                d_zone = int(self.rng.choice(self.n_zones, p=share))
            origin_lat = self.zone_lats[o_zone] + self.rng.normal(0, self.jitter_origin_deg)
            origin_lon = self.zone_lons[o_zone] + self.rng.normal(0, self.jitter_origin_deg)
            dest_lat = self.zone_lats[d_zone] + self.rng.normal(0, self.jitter_origin_deg)
            dest_lon = self.zone_lons[d_zone] + self.rng.normal(0, self.jitter_origin_deg)
            # فاصله مبدأ تا مقصد
            trip_dist = haversine_km(origin_lat, origin_lon, dest_lat, dest_lon)
            # اگر سفر خیلی کوتاه شد، سمپل را قبول کن همانطور است (سفرهای داخل ناحیه)

            pax = PassengerAgent(
                unique_id=self._gen_id(),
                model=self,
                origin_lat=origin_lat,
                origin_lon=origin_lon,
                dest_lat=dest_lat,
                dest_lon=dest_lon,
                origin_zone=int(o_zone),
                dest_zone=int(d_zone),
                request_time=self.schedule.steps,
                max_wait=sample_passenger_max_wait(self.rng),
                price_sensitivity=sample_price_sensitivity(self.rng),
                trip_distance_km=trip_dist,
            )
            self.passengers.append(pax)
            self.schedule.add(pax)
            self.total_requests += 1

        return n_new

    # ------------------------------------------------------------------
    # هوا
    # ------------------------------------------------------------------

    def _sample_weather_for_step(self, hour: int) -> str:
        """
        نمونه‌گیری هوای فعلی. هوا به‌صورت ساعتی (نه گامی) تغییر می‌کند تا
        پایداری منطقی داشته باشد. فعلاً ابتدای هر ساعت سمپل می‌شود.
        """
        # تغییر هوا در شروع هر ساعت
        minutes_in_hour = (self.schedule.steps * self.step_minutes) % 60
        if minutes_in_hour < self.step_minutes:
            raw = self.rng.choice(self._weather_labels_raw, p=self._weather_probs)
            self.current_weather = map_weather(str(raw))
        return self.current_weather

    # ------------------------------------------------------------------
    # حلقه اصلی گام
    # ------------------------------------------------------------------

    def step(self) -> None:
        """
        یک گام شبیه‌سازی (طبق الگوریتم ۱ فصل ۳).

        ۱) به‌روزرسانی شیفت رانندگان
        ۲) به‌روزرسانی هوا
        ۳) تولید درخواست‌های جدید (پواسون ناهمگن)
        ۴) قانون لغو (مسافران waiting با wait_time > max_wait)
        ۵) تخصیص ترتیبی (Dispatcher + State Refresh)
        ۶) به‌روزرسانی موقعیت رانندگان (en_route → busy → available)
        ۷) محاسبه KPIها
        ۸) DataCollector
        """
        hour = self.current_hour()

        # 1) به‌روزرسانی شیفت
        for d in self.drivers:
            d.update_shift_status(hour)

        # 2) هوا
        self._sample_weather_for_step(hour)

        # 3) تولید درخواست (مسافر.step() افزایش wait_time در schedule انجام می‌شود)
        n_new = self._generate_requests(hour)
        self.step_n_new_requests = n_new

        # افزایش wait_time برای مسافران waiting موجود — schedule.step() این را می‌کند
        # اما schedule فقط agentهایی که در schedule هستند را step می‌کند، و dispatcher را هم.
        # ما wait_time را اینجا دستی پیش از تخصیص آپدیت می‌کنیم تا منطق روشن باشد:
        for p in self.passengers:
            if p.status == PassengerStatus.WAITING and p.request_time < self.schedule.steps:
                p.wait_time += self.step_minutes

        # 4) قانون لغو
        self.step_n_cancelled = 0
        for p in self.passengers:
            if p.check_cancellation():
                self.step_n_cancelled += 1
                self.total_cancelled += 1

        # 5) جذابیت ناحیه و تخصیص ترتیبی
        self._zone_attractiveness = self._compute_zone_attractiveness(hour)
        n_assign, n_reject = self.dispatcher.dispatch_step(
            drivers=self.drivers,
            passengers=self.passengers,
            hour=hour,
            weather=self.current_weather,
            surge_multiplier=self.current_surge,
            zone_attractiveness=self._zone_attractiveness,
            base_speed_kmh=self.base_speed_kmh,
        )
        self.step_n_assignments = n_assign

        # 5.1) شناسایی no_driver:
        # مسافری که زمان درخواستش پایین آمده، waiting است، اما کاندیدی نداشت یا
        # همه کاندیدها ردش کردند (reject_count > 0 و هنوز در waiting است)
        # به طور دقیق، در این گام نشانه‌گذاری می‌کنیم: مسافرانی که از reject اول گذشتند
        # و هنوز waiting هستند، علامت‌گذاری ضمنی می‌شوند. خود no_driver نهایی در پایان شیفت
        # یا بر اساس expire ثبت می‌شود.

        # 6) پیشروی رانندگان و ثبت تکمیل سفر
        self.step_n_completed = 0
        for d in self.drivers:
            event = d.advance(self.step_minutes)
            if event == "picked_up":
                if d.current_passenger_id is not None:
                    pax = self._find_passenger(d.current_passenger_id)
                    if pax is not None:
                        pax.mark_in_trip()
            elif event == "completed":
                # شناسایی مسافر نهایی — مسافری که قبلاً in_trip بود و الان complete است
                # نکته: d.current_passenger_id بعد از completed به None شده. باید بازیابی کنیم.
                # روش ساده: تمام مسافران in_trip که driver_id آن‌ها این راننده است → completed.
                for pax in self.passengers:
                    if (pax.status == PassengerStatus.IN_TRIP
                            and pax.assigned_driver_id == d.driver_id):
                        pax.mark_completed()
                        self.step_n_completed += 1
                        self.total_completed += 1
                        break

        # 7) محاسبه KPIها
        self._compute_step_kpis()

        # 8) advance schedule
        self.schedule.step()

        # 9) DataCollector
        self.datacollector.collect(self)

        # gc: حذف مسافران تمام‌شده/لغوشده از لیست فعال (اختیاری برای حافظه)
        # فعلاً نگه می‌داریم تا KPI تجمعی سالم باشد.

    def _find_passenger(self, pid: int) -> Optional[PassengerAgent]:
        """جستجوی مسافر بر اساس شناسه (در حال حاضر خطی؛ کافی برای N متعارف)."""
        for p in self.passengers:
            if p.unique_id == pid:
                return p
        return None

    def _compute_step_kpis(self) -> None:
        """محاسبه WT, DU, CR برای ثبت در DataCollector."""
        # WT_step: میانگین total_wait_time (انتظار+pickup_eta) برای مسافران
        # که در این گام تازه به ASSIGNED تبدیل شدند.
        # شناسایی: assigned_driver_id != None و wait_time == 0 یا کوچک، و
        # request_time + delay ≈ steps فعلی. ساده‌ترین راه: ست کردن یک پرچم
        # «تازه تخصیص یافته» در dispatcher. اینجا تقریب: نگاه به مسافرانی که
        # status=ASSIGNED با wait_time ≤ step_minutes (تازه تخصیص یافته).
        cur = self.schedule.steps
        new_assigned_total_waits: List[float] = []
        for p in self.passengers:
            if (p.status == PassengerStatus.ASSIGNED
                    and p.assigned_driver_id is not None
                    and p.pickup_eta_min > 0   # یعنی assign_to فراخوانی شده
                    and (cur - p.request_time) * self.step_minutes - p.wait_time < self.step_minutes + 1e-6
                    and p.complete_time is None
                    and p.wait_time <= self.step_minutes * 1.5):
                # این مسافر در یک یا دو گام اخیر تخصیص یافته
                new_assigned_total_waits.append(p.total_wait_time)
        self.step_wt_minutes = (
            float(np.mean(new_assigned_total_waits)) if new_assigned_total_waits else 0.0
        )

        # DU: نسبت رانندگانی که busy یا en_route هستند به online
        n_busy = sum(1 for d in self.drivers
                     if d.status in (DriverStatus.BUSY, DriverStatus.EN_ROUTE))
        n_online = sum(1 for d in self.drivers
                       if d.status != DriverStatus.OFFLINE)
        self.step_du = float(n_busy / n_online) if n_online > 0 else 0.0

        # CR: نسبت کل تکمیل‌شده‌ها به کل درخواست‌ها تا الان
        if self.total_requests > 0:
            self.step_cr = float(self.total_completed / self.total_requests)
        else:
            self.step_cr = 0.0

    # ------------------------------------------------------------------
    # خلاصه اپیزود
    # ------------------------------------------------------------------

    def episode_summary(self) -> Dict[str, Any]:
        """خلاصه نهایی KPIها در پایان اپیزود."""
        # WT کل: میانگین total_wait_time (queue + pickup_eta) روی مسافران completed
        completed_waits = [
            p.total_wait_time for p in self.passengers
            if p.status == PassengerStatus.COMPLETED
        ]
        mean_wt = float(np.mean(completed_waits)) if completed_waits else 0.0

        # DU کل: مجموع busy_time / مجموع active_time
        total_busy = sum(d.total_busy_time + d.total_pickup_time for d in self.drivers)
        total_active = sum(d.total_active_time for d in self.drivers)
        mean_du = float(total_busy / total_active) if total_active > 0 else 0.0

        # CR
        cr = (self.total_completed / self.total_requests) if self.total_requests > 0 else 0.0

        # نرخ لغو و no_driver
        cancel_rate = (self.total_cancelled / self.total_requests) if self.total_requests > 0 else 0.0

        # no_driver: مسافرانی که هیچ‌گاه assigned نشدند و الان waiting/no_driver/expired
        n_no_driver = sum(
            1 for p in self.passengers
            if p.status in (PassengerStatus.WAITING, PassengerStatus.NO_DRIVER)
            and p.assigned_driver_id is None
        )
        no_driver_rate = (n_no_driver / self.total_requests) if self.total_requests > 0 else 0.0

        return {
            "seed": self.seed_val,
            "total_requests": self.total_requests,
            "total_completed": self.total_completed,
            "total_cancelled": self.total_cancelled,
            "n_no_driver": n_no_driver,
            "CR": cr,
            "cancel_rate": cancel_rate,
            "no_driver_rate": no_driver_rate,
            "mean_WT_min": mean_wt,
            "mean_DU": mean_du,
        }

    def get_collected_data(self) -> pd.DataFrame:
        """دریافت داده‌های جمع‌آوری‌شده به‌صورت DataFrame."""
        return self.datacollector.get_model_vars_dataframe()


# ============================================================================
# تست مستقل
# ============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from utils import load_targets, load_zones, setup_logging

    setup_logging()

    base = Path(__file__).resolve().parents[2]
    targets = load_targets(base / "data" / "calibration" / "targets.json")
    zones_data = load_zones(base / "data" / "calibration" / "zones.json")

    config = {
        "n_drivers": 800,
        "step_minutes": 2.0,
        "max_steps": 60,   # ۱۲۰ دقیقه برای تست سریع
        "base_speed_kmh": 30.0,
        "d_max_km": 5.0,
        "seed": 42,
    }

    model = RideHailingModel(config=config, targets=targets, zones_data=zones_data)
    for t in range(config["max_steps"]):
        model.step()
        if (t + 1) % 10 == 0:
            logger.info("step %d: waiting=%d available=%d assigns_step=%d",
                        t + 1, model._n_waiting(),
                        model._n_available_drivers(), model.step_n_assignments)

    summary = model.episode_summary()
    logger.info("Quick test summary: %s", summary)
