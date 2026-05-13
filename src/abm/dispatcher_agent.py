"""
عامل اعزام‌کننده (DispatcherAgent) برای ABM تخصیص تاکسی آنلاین.

طبق فصل ۳ بخش ۳-۵-۱-ج، این عامل سیاست تخصیص را اجرا می‌کند.
معماری Strategy Pattern استفاده شده تا در مرحله ۷ بتوان DQN را
بدون تغییر در کد ABM، جایگزین Greedy کرد.

تخصیص ترتیبی (Sequential Assignment with State Refresh) طبق
الگوریتم ۱ فصل ۳ پیاده‌سازی شده است: در هر گام، چندین جفت
ممکن است انتخاب شوند، اما هر بار فقط یک جفت انتخاب می‌شود و
وضعیت قبل از انتخاب بعدی به‌روز می‌گردد.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
from mesa import Agent

from .utils import haversine_km

if TYPE_CHECKING:
    from .model import RideHailingModel
    from .driver_agent import DriverAgent
    from .passenger_agent import PassengerAgent

logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    """یک جفت کاندید (راننده، مسافر، فاصله pickup)."""
    driver: "DriverAgent"
    passenger: "PassengerAgent"
    pickup_dist_km: float

    def __repr__(self) -> str:
        return (f"Candidate(d={self.driver.driver_id}, p={self.passenger.unique_id}, "
                f"dist={self.pickup_dist_km:.2f}km)")


# ============================================================================
# Strategy Interface
# ============================================================================

class AssignmentStrategy(ABC):
    """
    رابط استراتژی تخصیص.

    این رابط طوری طراحی شده که در مرحله ۷، یک کلاس DQNStrategy می‌تواند
    بدون تغییر در کد ABM، جایگزین GreedyStrategy شود.
    """

    @abstractmethod
    def select(self,
               candidates: List[Candidate],
               state_vector: Optional[np.ndarray] = None,
               rng: Optional[np.random.Generator] = None) -> Optional[Candidate]:
        """
        انتخاب یک کاندید از لیست.

        ورودی:
            candidates: لیست کاندیدهای فعلی (مرتب نیست)
            state_vector: بردار حالت سراسری (برای DQN؛ Greedy نادیده می‌گیرد)
            rng: مولد عدد تصادفی برای تصمیم‌های احتمالی (مثل ε-greedy)

        خروجی:
            کاندید انتخاب‌شده، یا None اگر لیست خالی باشد.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """نام استراتژی برای logging و گزارش."""
        ...


# ============================================================================
# Greedy Nearest-Driver
# ============================================================================

class GreedyStrategy(AssignmentStrategy):
    """
    استراتژی حریصانه: نزدیک‌ترین راننده به مسافر را انتخاب می‌کند.

    این همان خط‌مبنای صنعتی (فصل ۳ بخش ۳-۸-۲) است.
    در فاز اولیه ABM (مرحله ۵)، dispatcher داخلی از این استفاده می‌کند.
    """

    @property
    def name(self) -> str:
        return "greedy"

    def select(self,
               candidates: List[Candidate],
               state_vector: Optional[np.ndarray] = None,
               rng: Optional[np.random.Generator] = None) -> Optional[Candidate]:
        if not candidates:
            return None
        # min بر اساس pickup_dist
        best = min(candidates, key=lambda c: c.pickup_dist_km)
        return best


# ============================================================================
# Random Baseline (برای مقایسه — مرحله ۸)
# ============================================================================

class RandomStrategy(AssignmentStrategy):
    """استراتژی تصادفی — کران پایین عملکرد (فصل ۳ بخش ۳-۸-۱)."""

    @property
    def name(self) -> str:
        return "random"

    def select(self,
               candidates: List[Candidate],
               state_vector: Optional[np.ndarray] = None,
               rng: Optional[np.random.Generator] = None) -> Optional[Candidate]:
        if not candidates:
            return None
        if rng is None:
            rng = np.random.default_rng()
        idx = int(rng.integers(0, len(candidates)))
        return candidates[idx]


# ============================================================================
# DispatcherAgent
# ============================================================================

class DispatcherAgent(Agent):
    """
    عامل اعزام‌کننده — تنها عامل تصمیم‌گیر در ABM.

    این عامل در هر گام:
        1) ماتریس کاندیدها را می‌سازد (فاصله ≤ 5 km)
        2) با استراتژی فعال، یک جفت انتخاب می‌کند
        3) راننده‌ی منتخب درباره‌ی پذیرش تصمیم می‌گیرد (رابطه ۳-۵)
        4) در صورت پذیرش: تخصیص اجرا می‌شود
        5) در صورت رد: جفت‌های شامل آن راننده حذف، مسافر در صف می‌ماند
        6) state refresh و تکرار تا اتمام کاندیدها

    ویژگی‌ها:
        strategy: استراتژی تخصیص (Greedy فعلاً، در مرحله ۷ → DQN)
        d_max_km: شعاع کاندیدی (پیش‌فرض ۵ km طبق فصل ۳)

    متریک‌های ثبت‌شده:
        n_assignments_step: تعداد تخصیص‌های موفق در گام جاری
        n_rejections_step: تعداد ردهای راننده در گام جاری
        n_no_driver_step: تعداد مسافرانی که هیچ کاندیدی نداشتند
    """

    def __init__(
        self,
        unique_id: int,
        model: "RideHailingModel",
        strategy: AssignmentStrategy,
        d_max_km: float = 5.0,
    ) -> None:
        super().__init__(unique_id, model)
        self.strategy: AssignmentStrategy = strategy
        self.d_max_km: float = float(d_max_km)

        # متریک‌های گام
        self.n_assignments_step: int = 0
        self.n_rejections_step: int = 0
        self.n_no_driver_step: int = 0

        # متریک‌های تجمعی
        self.total_assignments: int = 0
        self.total_rejections: int = 0
        self.total_no_driver: int = 0

    # ------------------------------------------------------------------
    # ساخت کاندیدها
    # ------------------------------------------------------------------

    def build_candidates(
        self,
        drivers: List["DriverAgent"],
        passengers: List["PassengerAgent"],
    ) -> List[Candidate]:
        """
        ماتریس کاندیدها: تمام جفت‌های (راننده available, مسافر waiting)
        با pickup_dist <= d_max_km.

        بهینه‌سازی: ابتدا با بردارسازی numpy فاصله را محاسبه می‌کنیم.
        """
        from .driver_agent import DriverStatus
        from .passenger_agent import PassengerStatus

        avail_drivers = [d for d in drivers if d.status == DriverStatus.AVAILABLE]
        wait_pax = [p for p in passengers if p.status == PassengerStatus.WAITING]

        if not avail_drivers or not wait_pax:
            return []

        d_lats = np.array([d.lat for d in avail_drivers], dtype=np.float64)
        d_lons = np.array([d.lon for d in avail_drivers], dtype=np.float64)
        p_lats = np.array([p.origin_lat for p in wait_pax], dtype=np.float64)
        p_lons = np.array([p.origin_lon for p in wait_pax], dtype=np.float64)

        # ماتریس فاصله (n_drivers, n_pax) — Haversine برداری ساده
        dist_matrix = self._pairwise_haversine(d_lats, d_lons, p_lats, p_lons)

        # فقط کاندیدهای زیر شعاع
        candidates: List[Candidate] = []
        mask = dist_matrix <= self.d_max_km
        idx_d, idx_p = np.where(mask)
        for i_d, i_p in zip(idx_d, idx_p):
            candidates.append(Candidate(
                driver=avail_drivers[i_d],
                passenger=wait_pax[i_p],
                pickup_dist_km=float(dist_matrix[i_d, i_p]),
            ))

        return candidates

    @staticmethod
    def _pairwise_haversine(lats1: np.ndarray, lons1: np.ndarray,
                            lats2: np.ndarray, lons2: np.ndarray) -> np.ndarray:
        """محاسبه فاصله Haversine بین دو مجموعه نقطه (m×n)."""
        R = 6371.0088
        l1 = np.radians(lats1)[:, None]
        l2 = np.radians(lats2)[None, :]
        dlat = l2 - l1
        dlon = (np.radians(lons2)[None, :] - np.radians(lons1)[:, None])
        a = np.sin(dlat / 2) ** 2 + np.cos(l1) * np.cos(l2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c

    # ------------------------------------------------------------------
    # حلقه تخصیص ترتیبی (الگوریتم ۱)
    # ------------------------------------------------------------------

    def dispatch_step(
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
        اجرای یک گام تخصیص ترتیبی.

        ورودی:
            drivers, passengers: لیست عوامل
            hour, weather, surge_multiplier: شرایط محیطی
            zone_attractiveness: آرایه (n_zones,) جذابیت هر ناحیه
            base_speed_kmh: سرعت پایه برای محاسبه زمان pickup

        خروجی:
            (n_assignments, n_rejections)
        """
        self.n_assignments_step = 0
        self.n_rejections_step = 0

        candidates = self.build_candidates(drivers, passengers)
        if not candidates:
            return 0, 0

        logger.debug("dispatch_step: initial candidates=%d", len(candidates))

        while candidates:
            # 1) انتخاب توسط استراتژی
            chosen = self.strategy.select(candidates, state_vector=None, rng=self.model.rng)
            if chosen is None:
                break

            # 2) محاسبه احتمال پذیرش
            dest_zone = chosen.passenger.dest_zone
            zone_attr = float(zone_attractiveness[dest_zone])
            accepted = chosen.driver.decide_accept(
                passenger=chosen.passenger,
                pickup_dist_km=chosen.pickup_dist_km,
                surge_multiplier=surge_multiplier,
                zone_attractiveness=zone_attr,
            )

            if accepted:
                # 3) اجرای تخصیص
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

                # حذف جفت‌های شامل این راننده و این مسافر
                candidates = [
                    c for c in candidates
                    if c.driver.driver_id != chosen.driver.driver_id
                    and c.passenger.unique_id != chosen.passenger.unique_id
                ]
            else:
                # 4) رد شد — این مسافر در صف می‌ماند، اما این *جفت* خاص دیگر تلاش نمی‌شود
                chosen.passenger.mark_rejected()
                self.n_rejections_step += 1
                self.total_rejections += 1

                # حذف فقط این جفت (نه همه‌ی جفت‌های راننده)
                # تا راننده بتواند مسافر دیگری را امتحان کند، و مسافر بتواند راننده دیگری
                candidates = [
                    c for c in candidates
                    if not (c.driver.driver_id == chosen.driver.driver_id
                            and c.passenger.unique_id == chosen.passenger.unique_id)
                ]

        # 5) مسافران بدون کاندید را no_driver علامت بزن
        # (فقط آن‌هایی که در ابتدا اصلاً کاندید نداشتند یا همه ردشان کردند)
        # برای سادگی، اینجا فقط آن‌هایی که reject_count بالا دارند را در model مدیریت می‌کنیم.
        # شناسایی no_driver نهایی در model.step انجام می‌شود.

        return self.n_assignments_step, self.n_rejections_step

    def step(self) -> None:
        """گام Mesa: dispatcher فعلاً در model.step مستقیماً فراخوانی می‌شود."""
        pass

    def __repr__(self) -> str:
        return (f"<Dispatcher strategy={self.strategy.name} "
                f"total_assigns={self.total_assignments} rejects={self.total_rejections}>")


# ============================================================================
# تست مستقل
# ============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from utils import setup_logging
    setup_logging()
    logger.info("DispatcherAgent module loaded successfully.")
    logger.info("Strategies available: GreedyStrategy, RandomStrategy")
    logger.info("(For DQN strategy, see Phase 7 implementation.)")
