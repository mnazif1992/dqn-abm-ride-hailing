"""
استراتژی Hungarian — تخصیص بهینه دسته‌ای (upper bound کلاسیک).

در هر گام، با کتابخانه scipy.optimize.linear_sum_assignment ماتریس
هزینه (فاصله pickup) حل می‌شود تا مجموع کل فاصله‌ی تخصیص‌ها کمینه
شود. این الگوریتم Kuhn-Munkres با پیچیدگی O(n³) است و یک upper bound
کلاسیک برای کیفیت تخصیص قبل از RL محسوب می‌شود.

تفاوت بنیادی با Greedy/Random:
    Hungarian به‌ذات batch است — همه waiting passengers و available drivers
    را در هر گام به‌صورت یکجا با هم تخصیص می‌دهد، در حالی که Greedy/Random
    ترتیبی عمل می‌کنند (هر بار یک جفت با state refresh). این تفاوت در
    گزارش پایان‌نامه (فصل ۵) باید صریح ذکر شود.

سیاست رد (rejection) — تأیید کاربر:
    اگر راننده‌ی منتخب در تخصیص بهینه، مسافر را رد کند، این جفت در
    این گام unassigned باقی می‌ماند (مسافر waiting، راننده available)
    و تا گام بعد منتظر می‌ماند. Hungarian دوباره حل نمی‌شود
    (نه re-solve، نه fallback به Greedy).
"""
from __future__ import annotations

import logging
from typing import List, Tuple, TYPE_CHECKING

import numpy as np
from scipy.optimize import linear_sum_assignment

if TYPE_CHECKING:
    from src.abm.driver_agent import DriverAgent
    from src.abm.passenger_agent import PassengerAgent

logger = logging.getLogger(__name__)

# هزینه برای جفت‌های خارج از شعاع — به اندازه کافی بزرگ که هرگز انتخاب نشوند
_INVALID_COST = 1e9


def hungarian_dispatch_step(
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
    اجرای یک گام تخصیص با Hungarian (Kuhn-Munkres).

    منطق:
        1) فیلتر drivers (available) و passengers (waiting)
        2) محاسبه ماتریس فاصله Haversine به‌صورت برداری
        3) جفت‌های خارج از d_max_km را با هزینه INF mask می‌کنیم
        4) حل بهینه‌ی Hungarian → مجموع فاصله‌ی تخصیص‌ها کمینه
        5) برای هر جفت بهینه (که در شعاع است):
           - راننده accept/reject را تصمیم می‌گیرد
           - اگر accept: تخصیص اجرا
           - اگر reject: جفت تا گام بعد unassigned (طبق تأیید کاربر)
        6) هیچ re-solve یا fallback انجام نمی‌شود

    خروجی:
        (n_assignments_step, n_rejections_step)
    """
    from src.abm.driver_agent import DriverStatus
    from src.abm.passenger_agent import PassengerStatus

    self.n_assignments_step = 0
    self.n_rejections_step = 0

    avail_drivers = [d for d in drivers if d.status == DriverStatus.AVAILABLE]
    wait_pax = [p for p in passengers if p.status == PassengerStatus.WAITING]

    if not avail_drivers or not wait_pax:
        return 0, 0

    # 1) ساخت ماتریس فاصله Haversine بین driver/passenger (n_drv × n_pax)
    d_lats = np.array([d.lat for d in avail_drivers], dtype=np.float64)
    d_lons = np.array([d.lon for d in avail_drivers], dtype=np.float64)
    p_lats = np.array([p.origin_lat for p in wait_pax], dtype=np.float64)
    p_lons = np.array([p.origin_lon for p in wait_pax], dtype=np.float64)

    # استفاده از همان helper که DispatcherAgent دارد
    dist_matrix = self._pairwise_haversine(d_lats, d_lons, p_lats, p_lons)

    # 2) جفت‌های خارج از شعاع → هزینه INF (هرگز انتخاب نشوند)
    cost = np.where(dist_matrix <= self.d_max_km, dist_matrix, _INVALID_COST)

    # 3) حل Hungarian — برای ماتریس‌های غیرمربعی، min(n_drv, n_pax) جفت برمی‌گرداند
    row_ind, col_ind = linear_sum_assignment(cost)

    # 4) پردازش هر تخصیص بهینه
    for i, j in zip(row_ind, col_ind):
        # جفت‌های نامعتبر (خارج شعاع) را رد کن
        if cost[i, j] >= _INVALID_COST:
            continue

        driver = avail_drivers[i]
        passenger = wait_pax[j]
        pickup_dist = float(dist_matrix[i, j])

        # 5) راننده تصمیم پذیرش می‌گیرد
        zone_attr = float(zone_attractiveness[passenger.dest_zone])
        accepted = driver.decide_accept(
            passenger=passenger,
            pickup_dist_km=pickup_dist,
            surge_multiplier=surge_multiplier,
            zone_attractiveness=zone_attr,
        )

        if accepted:
            pickup_eta = driver.assign_to(
                passenger=passenger,
                base_speed_kmh=base_speed_kmh,
                hour=hour,
                weather=weather,
            )
            passenger.mark_assigned(driver.driver_id)
            passenger.pickup_eta_min = pickup_eta
            self.n_assignments_step += 1
            self.total_assignments += 1
        else:
            # طبق سیاست تأیید‌شده: جفت تا گام بعد منتظر می‌ماند
            # هیچ re-solve یا fallback Greedy انجام نمی‌شود
            passenger.mark_rejected()
            self.n_rejections_step += 1
            self.total_rejections += 1

    return self.n_assignments_step, self.n_rejections_step
