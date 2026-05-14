"""
استراتژی Hungarian-Resolve — Hungarian با حل مجدد پس از رد.

تفاوت بنیادی با hungarian_baseline:
    در hungarian_baseline، اگر راننده رد کند، جفت تا گام بعد منتظر می‌ماند.
    در این نسخه، پس از هر دور حل، جفت‌های رد شده «در همان گام» مجدداً
    در حلقه‌ی بعدی Hungarian شرکت می‌کنند (با مسافر/راننده‌ی دیگری).
    این "Hungarian خالص نظری" است (گزینه C در سؤال طراحی اولیه).

منطق دقیق در هر iteration:
    1) حل Hungarian روی ماتریس هزینه فعلی
    2) برای هر جفت بهینه:
       - جفت accept → ردیف (driver) و ستون (passenger) از ماتریس INF می‌شوند
         (driver busy شد، passenger assigned شد — هر دو از pool خارج)
       - جفت reject → فقط همان سلول INF می‌شود
         (driver و passenger همچنان می‌توانند با دیگری امتحان شوند)
    3) اگر هیچ جفت معتبری پیدا نشد یا MAX_ITER رسید، حلقه پایان می‌یابد

پیش‌بینی نظری:
    باید عملکرد مشابه Greedy داشته باشد (هر دو می‌توانند retry کنند درون گام)،
    اما با مزیت گلوبال بودن انتخاب در هر iteration. اگر هنوز از Greedy ضعیف‌تر بود،
    یعنی مزیت گلوبال در حضور stochastic acceptance ضعیف است.

هزینه محاسباتی:
    O(n³ × MAX_ITER) به‌جای O(n³) — ولی در عمل، با INF شدن سلول‌ها،
    iterationهای بعدی بسیار سریع‌تر همگرا می‌شوند.
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

# هزینه برای جفت‌های نامعتبر (خارج شعاع یا rejected یا قبلاً تخصیص یافته)
_INVALID_COST = 1e9

# سقف ایمنی iteration در هر گام (جلوگیری از infinite loop در حالات بحرانی)
_MAX_ITERATIONS = 10


def hungarian_resolve_dispatch_step(
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
    اجرای یک گام تخصیص با Hungarian + re-solve پس از رد.

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

    # 1) ساخت ماتریس فاصله Haversine (n_drv × n_pax)
    d_lats = np.array([d.lat for d in avail_drivers], dtype=np.float64)
    d_lons = np.array([d.lon for d in avail_drivers], dtype=np.float64)
    p_lats = np.array([p.origin_lat for p in wait_pax], dtype=np.float64)
    p_lons = np.array([p.origin_lon for p in wait_pax], dtype=np.float64)

    dist_matrix = self._pairwise_haversine(d_lats, d_lons, p_lats, p_lons)

    # ماتریس هزینه قابل mutation: ابتدا جفت‌های خارج شعاع INF
    cost = np.where(dist_matrix <= self.d_max_km, dist_matrix, _INVALID_COST).copy()

    # 2) حلقه‌ی re-solve
    for iteration in range(_MAX_ITERATIONS):
        # اگر هیچ سلول معتبری باقی نمانده، تمام
        if not (cost < _INVALID_COST).any():
            break

        # حل Hungarian روی ماتریس فعلی
        row_ind, col_ind = linear_sum_assignment(cost)

        found_valid_in_iter = False
        for i, j in zip(row_ind, col_ind):
            # رد کردن جفت‌های نامعتبر (خارج شعاع یا قبلاً مسدود شده)
            if cost[i, j] >= _INVALID_COST:
                continue
            found_valid_in_iter = True

            driver = avail_drivers[i]
            passenger = wait_pax[j]
            pickup_dist = float(dist_matrix[i, j])

            # 3) راننده تصمیم پذیرش می‌گیرد
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
                # حذف کامل: driver busy شد، passenger assigned شد
                cost[i, :] = _INVALID_COST
                cost[:, j] = _INVALID_COST
            else:
                passenger.mark_rejected()
                self.n_rejections_step += 1
                self.total_rejections += 1
                # فقط این جفت خاص مسدود — دو طرف می‌توانند با دیگری امتحان شوند
                cost[i, j] = _INVALID_COST

        # اگر در این iteration هیچ جفت معتبری پیدا نشد، خروج
        if not found_valid_in_iter:
            break

    return self.n_assignments_step, self.n_rejections_step
