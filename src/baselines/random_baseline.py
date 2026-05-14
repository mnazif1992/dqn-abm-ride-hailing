"""
استراتژی Random — کران پایین (lower bound) عملکرد.

در هر گام، از کاندیدهای موجود (راننده‌های available در شعاع d_max_km)،
یک جفت به‌صورت تصادفی یکنواخت انتخاب می‌شود. سپس راننده با احتمال
acceptance (نمونه‌گیری شده از Beta در زمان initialization) تصمیم به
پذیرش/رد می‌گیرد.

کاربرد:
    این استراتژی به‌عنوان lower bound عملکرد در مقایسه با Greedy/Hungarian
    و DQN استفاده می‌شود. اگر DQN از Random بهتر نباشد، یعنی یادگیری
    صورت نگرفته است (sanity check).

تفاوت با Greedy:
    - Greedy: انتخاب نزدیک‌ترین راننده به مسافر
    - Random: انتخاب تصادفی یکنواخت — بدون توجه به فاصله
"""
from __future__ import annotations

import logging
from typing import List, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.abm.driver_agent import DriverAgent
    from src.abm.passenger_agent import PassengerAgent

logger = logging.getLogger(__name__)


def random_dispatch_step(
    self,
    drivers: List["DriverAgent"],
    passengers: List["PassengerAgent"],
    hour: int,
    weather: str,
    surge_multiplier: float,
    zone_attractiveness: np.ndarray,
    base_speed_kmh: float,
) -> Tuple[int, int]:
    """
    اجرای یک گام تخصیص تصادفی.

    منطق:
        1) ساخت کاندیدها (drivers available در شعاع d_max_km)
        2) از کاندیدها یکی را تصادفی یکنواخت انتخاب کن
        3) راننده accept/reject را تصمیم می‌گیرد (مدل رفتاری ABM)
        4) اگر accept: تخصیص اجرا، حذف همه جفت‌های شامل این driver/passenger
        5) اگر reject: فقط این جفت حذف، بقیه‌ی جفت‌ها همچنان قابل امتحان
        6) تکرار تا اتمام کاندیدها

    خروجی:
        (n_assignments_step, n_rejections_step)
    """
    self.n_assignments_step = 0
    self.n_rejections_step = 0

    candidates = self.build_candidates(drivers, passengers)
    if not candidates:
        return 0, 0

    rng = self.model.rng

    while candidates:
        # 1) انتخاب تصادفی یکنواخت از باقیمانده کاندیدها
        idx = int(rng.integers(0, len(candidates)))
        chosen = candidates[idx]

        # 2) تصمیم پذیرش راننده
        zone_attr = float(zone_attractiveness[chosen.passenger.dest_zone])
        accepted = chosen.driver.decide_accept(
            passenger=chosen.passenger,
            pickup_dist_km=chosen.pickup_dist_km,
            surge_multiplier=surge_multiplier,
            zone_attractiveness=zone_attr,
        )

        if accepted:
            # 3.a) تخصیص موفق
            pickup_eta = chosen.driver.assign_to(
                passenger=chosen.passenger,
                base_speed_kmh=base_speed_kmh,
                hour=hour,
                weather=weather,
            )
            chosen.passenger.mark_assigned(chosen.driver.driver_id)
            chosen.passenger.pickup_eta_min = pickup_eta
            self.n_assignments_step += 1
            self.total_assignments += 1

            # حذف همه جفت‌های شامل این driver یا این passenger
            candidates = [
                c for c in candidates
                if c.driver.driver_id != chosen.driver.driver_id
                and c.passenger.unique_id != chosen.passenger.unique_id
            ]
        else:
            # 3.b) راننده رد کرد — مسافر در صف می‌ماند، راننده هم
            chosen.passenger.mark_rejected()
            self.n_rejections_step += 1
            self.total_rejections += 1

            # حذف فقط این جفت خاص — دو طرف می‌توانند با دیگری امتحان کنند
            candidates = [
                c for c in candidates
                if not (c.driver.driver_id == chosen.driver.driver_id
                        and c.passenger.unique_id == chosen.passenger.unique_id)
            ]

    return self.n_assignments_step, self.n_rejections_step
