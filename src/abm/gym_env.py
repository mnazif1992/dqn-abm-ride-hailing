"""
محیط Gymnasium حول ABM تخصیص تاکسی آنلاین (مرحله ۶).

این ماژول کلاس RideHailingEnv را تعریف می‌کند که شبیه‌سازی ABM موجود
(RideHailingModel در src/abm/model.py) را در قالب استاندارد gymnasium.Env
می‌پوشاند تا عامل DQN بتواند با آن تعامل کند. هیچ تغییری در src/abm/
داده نمی‌شود؛ کنترل تخصیص با monkey-patching موقت dispatch_step به no-op
به‌دست محیط سپرده می‌شود (مشابه الگوی مراحل ۴ و ۵).

نگاشت به فصل ۳ پایان‌نامه:
    - فضای حالت S (بخش ۳-۴-۲ الف، جدول ۳-۳): بردار ۳۲ بُعدی
        ۴ مؤلفه زمانی (hour_sin/cos, day_sin/cos)
        ۱۴ مؤلفه ناحیه (zone_id one-hot)
        ۴ مؤلفه تقاضا/عرضه (demand_supply_index, surge,
                            n_available_drivers, n_pending_requests)
        ۳ مؤلفه عملکرد (mean_WT_zone, mean_pickup_dist, DU_zone)
        ۴ مؤلفه آب‌وهوا (clear/cloudy/rainy/heavy_rain one-hot)
        ۳ مؤلفه پرچم (is_weekend, is_holiday, is_rush_hour)
        مجموع = ۴+۱۴+۴+۳+۴+۳ = ۳۲
    - فضای اقدام A (بخش ۳-۴-۲ ب): انتخاب یک جفت کاندید (driver, passenger)
      با معماری Q(s, a_features). هر کاندید ۸ ویژگی دارد (جدول ۳-۴).
      اینجا به‌صورت Discrete(K_MAX+1) پیاده شده — اندیس ۰..K_MAX-1 انتخاب
      کاندید، اندیس K_MAX = no-op (پایان دور تخصیص این گام زمانی).
    - تابع پاداش جزئی (رابطه ۳-۸):
        r_step = −α·(pickup_eta / WT_baseline)
                 + β·(1 / N_drivers_zone)
                 − λ·𝟙[reject]
    - تعریف اپیزود: T=۷۲۰ گام ABM (هر گام ۲ دقیقه) = ۲۴ ساعت = ۱ روز.

پروتکل گذار (بخش ۳-۷-۲): هر فراخوانی step() یک گذار مستقل تولید می‌کند.
وقتی کاندیدهای یک گام زمانی تمام شد، زمان ABM یک گام جلو می‌رود.

مرجع: فصل ۳، بخش‌های ۳-۴ تا ۳-۸، جدول‌های ۳-۳ و ۳-۴، الگوریتم‌های ۱ و ۲.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "gymnasium لازم است: pip install gymnasium"
    ) from exc

logger = logging.getLogger(__name__)


def _noop_dispatch_step(
    self,
    drivers: List[Any],
    passengers: List[Any],
    hour: int,
    weather: str,
    surge_multiplier: float,
    zone_attractiveness: np.ndarray,
    base_speed_kmh: float,
) -> Tuple[int, int]:
    """
    جایگزین no-op برای DispatcherAgent.dispatch_step.

    وقتی محیط Gym کنترل تخصیص را به‌دست می‌گیرد، model.step() باید فقط
    تولید درخواست، لغوها، پیشروی سفرها و محاسبه KPI را انجام دهد و
    خودش هیچ تخصیصی نکند. خود تخصیص توسط RideHailingEnv.step انجام می‌شود.
    """
    self.n_assignments_step = 0
    self.n_rejections_step = 0
    return 0, 0


class RideHailingEnv(gym.Env):
    """
    محیط RL برای مسئله تخصیص پویای تاکسی آنلاین (نگاشت ABM → MDP).

    هر گام محیط (step) معادل یک تصمیم تخصیص برای یک جفت کاندید است
    (تخصیص ترتیبی با state refresh، الگوریتم ۱ فصل ۳). وقتی کاندیدهای
    گام زمانی جاری تمام شد یا عامل no-op داد، یک گام زمانی ABM (۲ دقیقه)
    پیش می‌رود.
    """

    metadata = {"render_modes": ["human"]}

    K_MAX: int = 50            # حداکثر تعداد کاندید در هر تصمیم
    STATE_DIM: int = 32        # بُعد بردار حالت (جدول ۳-۳)
    ACTION_FEAT_DIM: int = 8   # بُعد ویژگی هر کاندید (جدول ۳-۴)

    # ترتیب one-hot آب‌وهوا (سازگار با ABM/کالیبراسیون)
    _WEATHER_ORDER = ("clear", "cloudy", "rainy", "heavy_rain")

    def __init__(
        self,
        config: Dict[str, Any],
        targets: Dict[str, Any],
        zones_data: Dict[str, Any],
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        self._config = dict(config)
        self._targets = dict(targets)
        self._zones_data = dict(zones_data)
        self._base_seed = seed
        self.render_mode = render_mode

        # ضرایب پاداش از بخش reward در abm_calibrated.yaml (با fallback فصل ۳)
        reward_cfg = dict(self._config.get("reward", {}))
        self.alpha: float = float(reward_cfg.get("alpha", 0.6))
        self.beta: float = float(reward_cfg.get("beta", 0.4))
        self.lambda_reject: float = float(reward_cfg.get("lambda_reject", 0.2))
        self.WT_baseline: float = float(
            reward_cfg.get("WT_baseline", self._targets.get("WT_baseline", 2.5154))
        )
        self.DU_baseline: float = float(
            reward_cfg.get("DU_baseline", self._targets.get("DU_baseline", 0.6364))
        )

        # افق زمانی اپیزود (T=۷۲۰ گام × ۲ دقیقه = ۲۴ ساعت)
        self.max_steps: int = int(self._config.get("max_steps", 720))
        self.n_zones: int = 14

        # ثابت‌های نرمال‌سازی ویژگی‌ها
        self._d_max_km: float = float(self._config.get("d_max_km", 5.0))
        self._trip_dist_max_km: float = 50.0
        self._eta_max_min: float = 60.0
        self._wait_max_min: float = 30.0
        self._fare_max_toman: float = float(
            self._targets.get("fare_mean_toman", 38610.0)
        ) * 5.0

        # --- فضای مشاهده: Dict مطابق درخواست ---
        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(
                    low=-1.0, high=1.0,
                    shape=(self.STATE_DIM,), dtype=np.float32,
                ),
                "candidates": spaces.Box(
                    low=0.0, high=1.0,
                    shape=(self.K_MAX, self.ACTION_FEAT_DIM), dtype=np.float32,
                ),
                "candidate_mask": spaces.Box(
                    low=0, high=1,
                    shape=(self.K_MAX,), dtype=np.int8,
                ),
                "n_candidates": spaces.Discrete(self.K_MAX + 1),
            }
        )

        # --- فضای اقدام: انتخاب کاندید یا no-op (اندیس K_MAX) ---
        self.action_space = spaces.Discrete(self.K_MAX + 1)

        # وضعیت داخلی
        self._model: Any = None
        self._dispatcher: Any = None
        self._orig_dispatch_step: Any = None
        self._candidates: List[Any] = []
        self._episode_reward: float = 0.0
        self._abm_step_count: int = 0
        self._terminated: bool = False

    # ------------------------------------------------------------------
    # ساخت/مدیریت مدل ABM
    # ------------------------------------------------------------------

    def _patch_dispatch(self) -> None:
        """monkey-patch موقت dispatch_step به no-op (کنترل با محیط)."""
        from src.abm import dispatcher_agent

        if self._orig_dispatch_step is None:
            self._orig_dispatch_step = dispatcher_agent.DispatcherAgent.dispatch_step
        dispatcher_agent.DispatcherAgent.dispatch_step = _noop_dispatch_step

    def _unpatch_dispatch(self) -> None:
        """بازگرداندن dispatch_step اصلی ABM."""
        from src.abm import dispatcher_agent

        if self._orig_dispatch_step is not None:
            dispatcher_agent.DispatcherAgent.dispatch_step = self._orig_dispatch_step
            self._orig_dispatch_step = None

    def _build_model(self, seed: int) -> None:
        """ساخت یک نمونه تازه RideHailingModel با seed مشخص."""
        from src.abm.model import RideHailingModel

        cfg = dict(self._config)
        cfg["seed"] = int(seed)
        cfg["max_steps"] = self.max_steps
        self._model = RideHailingModel(
            config=cfg,
            targets=self._targets,
            zones_data=self._zones_data,
        )
        self._dispatcher = self._model.dispatcher
        self._abm_step_count = 0

    # ------------------------------------------------------------------
    # کاندیدها
    # ------------------------------------------------------------------

    def _collect_candidates(self) -> List[Any]:
        """
        ساخت لیست کاندیدهای فعلی (drivers available × waiting pax در شعاع).

        از build_candidates خودِ DispatcherAgent استفاده می‌شود؛ سپس بر
        اساس فاصله pickup مرتب و حداکثر K_MAX کاندید نگه داشته می‌شود.
        """
        cands = self._dispatcher.build_candidates(
            self._model.drivers, self._model.passengers
        )
        cands.sort(key=lambda c: c.pickup_dist_km)
        return cands[: self.K_MAX]

    def _advance_abm_time(self) -> None:
        """
        پیشروی یک گام زمانی ABM (۲ دقیقه).

        چون dispatch_step به no-op patch شده، model.step() فقط درخواست
        تولید می‌کند، لغوها را پردازش می‌کند، سفرها را پیش می‌برد و KPI
        محاسبه می‌کند — بدون تخصیص خودکار.
        """
        self._model.step()
        self._abm_step_count += 1

    # ------------------------------------------------------------------
    # بردار حالت ۳۲ بُعدی (جدول ۳-۳)
    # ------------------------------------------------------------------

    def _active_zone(self) -> int:
        """
        ناحیه فعال = ناحیه با بیشترین درخواست معلق (round-robin ساده).

        مطابق بخش ۳-۷-۱: state در سطح ناحیه فعال محاسبه می‌شود.
        """
        from src.abm.passenger_agent import PassengerStatus

        counts = np.zeros(self.n_zones, dtype=np.int64)
        for p in self._model.passengers:
            if p.status == PassengerStatus.WAITING:
                z = int(getattr(p, "origin_zone", 0))
                if 0 <= z < self.n_zones:
                    counts[z] += 1
        if counts.sum() == 0:
            return 0
        return int(np.argmax(counts))

    def _compute_state_vector(self) -> np.ndarray:
        """ساخت بردار حالت ۳۲ بُعدی نرمال‌شده طبق جدول ۳-۳."""
        from src.abm.driver_agent import DriverStatus
        from src.abm.passenger_agent import PassengerStatus

        m = self._model
        step_idx = int(getattr(m.schedule, "steps", self._abm_step_count))
        step_minutes = float(getattr(m, "step_minutes", 2.0))

        # دقیقه/ساعت/روز شبیه‌سازی
        total_minutes = step_idx * step_minutes
        hour_of_day = int((total_minutes // 60) % 24)
        day_idx = int((total_minutes // (60 * 24)))
        dow = day_idx % 7

        # ۴ مؤلفه زمانی (سینوسی-کسینوسی)
        hour_sin = np.sin(2 * np.pi * hour_of_day / 24.0)
        hour_cos = np.cos(2 * np.pi * hour_of_day / 24.0)
        day_sin = np.sin(2 * np.pi * dow / 7.0)
        day_cos = np.cos(2 * np.pi * dow / 7.0)

        # ۱۴ مؤلفه ناحیه (one-hot ناحیه فعال)
        zone = self._active_zone()
        zone_onehot = np.zeros(self.n_zones, dtype=np.float32)
        zone_onehot[zone] = 1.0

        # شمارش‌های سراسری/ناحیه‌ای
        avail = [d for d in m.drivers if d.status == DriverStatus.AVAILABLE]
        waiting = [p for p in m.passengers if p.status == PassengerStatus.WAITING]
        n_drivers_total = max(1, len(m.drivers))
        n_avail = len(avail)
        n_pending = len(waiting)

        # ۴ مؤلفه تقاضا/عرضه
        demand_supply_index = float(
            np.clip(n_pending / max(1, n_avail), 0.0, 1.0)
        )
        surge = float(np.clip(getattr(m, "current_surge", 1.0) / 3.0, 0.0, 1.0))
        n_avail_norm = float(np.clip(n_avail / n_drivers_total, 0.0, 1.0))
        n_pending_norm = float(np.clip(n_pending / n_drivers_total, 0.0, 1.0))

        # ۳ مؤلفه عملکرد
        zone_waits = [
            float(getattr(p, "wait_time", 0.0))
            for p in waiting
            if int(getattr(p, "origin_zone", -1)) == zone
        ]
        mean_wt_zone = float(
            np.clip(np.mean(zone_waits) / self._wait_max_min, 0.0, 1.0)
        ) if zone_waits else 0.0

        cur_cands = self._candidates if self._candidates else []
        mean_pickup = float(
            np.clip(
                np.mean([c.pickup_dist_km for c in cur_cands]) / self._d_max_km,
                0.0, 1.0,
            )
        ) if cur_cands else 0.0

        n_busy = sum(
            1 for d in m.drivers
            if d.status in (DriverStatus.BUSY, DriverStatus.EN_ROUTE)
        )
        n_online = sum(
            1 for d in m.drivers if d.status != DriverStatus.OFFLINE
        )
        du_zone = float(np.clip(n_busy / n_online, 0.0, 1.0)) if n_online else 0.0

        # ۴ مؤلفه آب‌وهوا (one-hot)
        weather_raw = str(getattr(m, "current_weather", "clear")).lower()
        weather_onehot = np.zeros(4, dtype=np.float32)
        if weather_raw in self._WEATHER_ORDER:
            weather_onehot[self._WEATHER_ORDER.index(weather_raw)] = 1.0
        else:
            weather_onehot[0] = 1.0  # پیش‌فرض clear

        # ۳ مؤلفه پرچم
        is_weekend = 1.0 if dow in (3, 4) else 0.0  # پنجشنبه/جمعه ایران
        is_holiday = 0.0  # تعطیلات رسمی در ABM فعلی مدل نشده
        is_rush = 1.0 if hour_of_day in (7, 8, 9, 17, 18, 19, 20) else 0.0

        state = np.concatenate(
            [
                np.array([hour_sin, hour_cos, day_sin, day_cos], dtype=np.float32),
                zone_onehot,
                np.array(
                    [demand_supply_index, surge, n_avail_norm, n_pending_norm],
                    dtype=np.float32,
                ),
                np.array([mean_wt_zone, mean_pickup, du_zone], dtype=np.float32),
                weather_onehot,
                np.array([is_weekend, is_holiday, is_rush], dtype=np.float32),
            ]
        ).astype(np.float32)

        assert state.shape[0] == self.STATE_DIM, (
            f"state dim {state.shape[0]} != {self.STATE_DIM}"
        )
        return state

    # ------------------------------------------------------------------
    # ویژگی‌های کاندید ۸ بُعدی (جدول ۳-۴)
    # ------------------------------------------------------------------

    def _candidate_features(self, cand: Any) -> np.ndarray:
        """بردار ۸ ویژگی نرمال‌شده برای یک جفت (driver, passenger)."""
        m = self._model
        d = cand.driver
        p = cand.passenger
        pickup_dist = float(cand.pickup_dist_km)

        # 1) pickup_distance
        f_pickup = np.clip(pickup_dist / self._d_max_km, 0.0, 1.0)

        # 2) trip_distance
        trip_dist = float(getattr(p, "trip_distance_km", 0.0))
        f_trip = np.clip(trip_dist / self._trip_dist_max_km, 0.0, 1.0)

        # 3) estimated_eta (دقیقه) = فاصله / سرعت پایه × ۶۰
        base_speed = float(getattr(m, "base_speed_kmh", 30.0))
        eta_min = (pickup_dist / max(base_speed, 1e-6)) * 60.0
        f_eta = np.clip(eta_min / self._eta_max_min, 0.0, 1.0)

        # 4) driver_acceptance_rate
        f_acc = np.clip(float(getattr(d, "acceptance_rate", 0.8)), 0.0, 1.0)

        # 5) driver_utilization_so_far
        f_util = np.clip(
            float(getattr(d, "utilization_so_far",
                          getattr(d, "driver_utilization", 0.0))),
            0.0, 1.0,
        )

        # 6) destination_zone_attractiveness
        dest_zone = int(getattr(p, "dest_zone", 0))
        zattr = getattr(m, "_zone_attractiveness", None)
        if zattr is not None and 0 <= dest_zone < len(zattr):
            f_dest = np.clip(float(zattr[dest_zone]), 0.0, 1.0)
        else:
            f_dest = 0.5

        # 7) passenger_wait_so_far
        f_wait = np.clip(
            float(getattr(p, "wait_time", 0.0)) / self._wait_max_min, 0.0, 1.0
        )

        # 8) fare_estimate (تخمین بر اساس فاصله سفر)
        fare = float(getattr(p, "fare_toman", trip_dist * 4000.0))
        f_fare = np.clip(fare / max(self._fare_max_toman, 1e-6), 0.0, 1.0)

        return np.array(
            [f_pickup, f_trip, f_eta, f_acc, f_util, f_dest, f_wait, f_fare],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # ساخت observation
    # ------------------------------------------------------------------

    def _build_observation(self) -> Dict[str, Any]:
        """مونتاژ observation از حالت + ماتریس کاندیدها + ماسک."""
        state = self._compute_state_vector()

        cand_mat = np.zeros(
            (self.K_MAX, self.ACTION_FEAT_DIM), dtype=np.float32
        )
        mask = np.zeros(self.K_MAX, dtype=np.int8)
        n = min(len(self._candidates), self.K_MAX)
        for i in range(n):
            cand_mat[i] = self._candidate_features(self._candidates[i])
            mask[i] = 1

        return {
            "state": state,
            "candidates": cand_mat,
            "candidate_mask": mask,
            "n_candidates": n,
        }

    # ------------------------------------------------------------------
    # پاداش جزئی (رابطه ۳-۸)
    # ------------------------------------------------------------------

    def _partial_reward(
        self,
        pickup_eta_min: float,
        n_drivers_zone: int,
        rejected: bool,
    ) -> float:
        """
        r_step = −α·(pickup_eta / WT_baseline)
                 + β·(1 / N_drivers_zone)
                 − λ·𝟙[reject]
        """
        n_zone = max(1, int(n_drivers_zone))
        r = (
            -self.alpha * (pickup_eta_min / max(self.WT_baseline, 1e-6))
            + self.beta * (1.0 / n_zone)
            - self.lambda_reject * (1.0 if rejected else 0.0)
        )
        return float(r)

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """شروع یک اپیزود تازه (یک روز کامل = ۷۲۰ گام)."""
        super().reset(seed=seed)
        episode_seed = (
            int(seed) if seed is not None
            else (int(self._base_seed) if self._base_seed is not None else 42)
        )

        self._patch_dispatch()
        self._build_model(episode_seed)
        self._episode_reward = 0.0
        self._terminated = False

        # گام اول: تولید درخواست‌های اولیه (بدون تخصیص چون dispatch=no-op)
        self._advance_abm_time()
        self._candidates = self._collect_candidates()

        # اگر گام اول کاندید نداشت، تا یافتن کاندید یا پایان اپیزود جلو برو
        while not self._candidates and self._abm_step_count < self.max_steps:
            self._advance_abm_time()
            self._candidates = self._collect_candidates()

        obs = self._build_observation()
        info: Dict[str, Any] = {
            "abm_step": self._abm_step_count,
            "n_candidates": len(self._candidates),
            "seed": episode_seed,
        }
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        یک تصمیم تخصیص. action ∈ [0, K_MAX]:
            - 0..K_MAX-1: انتخاب کاندید با همان اندیس
            - K_MAX: no-op (پایان دور تخصیص این گام زمانی → پیشروی زمان)
        """
        from src.abm.driver_agent import DriverStatus

        if self._terminated:
            raise RuntimeError("step() پس از پایان اپیزود؛ ابتدا reset() کنید.")

        action = int(action)
        reward = 0.0
        info: Dict[str, Any] = {}

        n_cur = len(self._candidates)
        is_noop = (action >= self.K_MAX) or (action >= n_cur)

        if is_noop:
            # پایان دور تخصیص این گام زمانی → پیشروی زمان ABM
            info["action_type"] = "noop"
        else:
            cand = self._candidates[action]
            d = cand.driver
            p = cand.passenger
            pickup_dist = float(cand.pickup_dist_km)

            # تصمیم پذیرش راننده (مدل رفتاری ABM، رابطه ۳-۵)
            zattr = getattr(self._model, "_zone_attractiveness", None)
            dest_zone = int(getattr(p, "dest_zone", 0))
            if zattr is not None and 0 <= dest_zone < len(zattr):
                zone_attr = float(zattr[dest_zone])
            else:
                zone_attr = 0.5

            accepted = bool(
                d.decide_accept(
                    passenger=p,
                    pickup_dist_km=pickup_dist,
                    surge_multiplier=float(
                        getattr(self._model, "current_surge", 1.0)
                    ),
                    zone_attractiveness=zone_attr,
                )
            )

            n_avail_zone = sum(
                1 for dd in self._model.drivers
                if dd.status == DriverStatus.AVAILABLE
            )

            if accepted:
                pickup_eta = float(
                    d.assign_to(
                        passenger=p,
                        base_speed_kmh=float(
                            getattr(self._model, "base_speed_kmh", 30.0)
                        ),
                        hour=int(
                            (self._abm_step_count
                             * getattr(self._model, "step_minutes", 2.0)
                             // 60) % 24
                        ),
                        weather=str(
                            getattr(self._model, "current_weather", "clear")
                        ),
                    )
                )
                p.mark_assigned(d.driver_id)
                p.pickup_eta_min = pickup_eta
                self._dispatcher.total_assignments += 1
                reward = self._partial_reward(
                    pickup_eta_min=pickup_eta,
                    n_drivers_zone=n_avail_zone,
                    rejected=False,
                )
                info["action_type"] = "assigned"
            else:
                p.mark_rejected()
                self._dispatcher.total_rejections += 1
                reward = self._partial_reward(
                    pickup_eta_min=0.0,
                    n_drivers_zone=n_avail_zone,
                    rejected=True,
                )
                info["action_type"] = "rejected"

            # حذف جفت‌های شامل این driver یا این passenger
            self._candidates = [
                c for c in self._candidates
                if c.driver.driver_id != d.driver_id
                and c.passenger.unique_id != p.unique_id
            ]

        # اگر کاندید تمام شد یا no-op شد → زمان ABM جلو می‌رود
        if is_noop or not self._candidates:
            while True:
                if self._abm_step_count >= self.max_steps:
                    self._terminated = True
                    break
                self._advance_abm_time()
                self._candidates = self._collect_candidates()
                if self._candidates or self._abm_step_count >= self.max_steps:
                    if not self._candidates:
                        self._terminated = True
                    break

        self._episode_reward += reward

        terminated = self._terminated
        truncated = False

        if terminated:
            try:
                summary = self._model.episode_summary()
            except Exception:  # pragma: no cover
                summary = {}
            info["episode_summary"] = summary
            info["episode_reward"] = self._episode_reward

        obs = self._build_observation()
        info["abm_step"] = self._abm_step_count
        info["n_candidates"] = len(self._candidates)
        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        """نمایش متنی ساده وضعیت جاری (render_mode='human')."""
        if self.render_mode != "human":
            return
        print(
            f"[RideHailingEnv] abm_step={self._abm_step_count}/{self.max_steps} "
            f"candidates={len(self._candidates)} "
            f"episode_reward={self._episode_reward:.3f}"
        )

    def close(self) -> None:
        """آزادسازی منابع و بازگرداندن dispatch_step اصلی ABM."""
        self._unpatch_dispatch()
        self._model = None
        self._dispatcher = None
        self._candidates = []
