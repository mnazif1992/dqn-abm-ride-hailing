"""
V-greedy decision maker for ride-hailing dispatch.

Uses V-network to compute edge weights and selects argmax candidate.
Reference: Tang et al. 2019 (KDD), Qin et al. 2020 (INFORMS).
"""

import numpy as np
import torch
from typing import List, Tuple, Optional

from src.dqn.v_network import VNetwork


class VGreedyPolicy:
    """
    Per-candidate V-greedy decision maker.

    For each candidate driver, compute:
        edge_weight = (1.0 - alpha * pickup_km / pickup_norm)
                    + gamma * V(s_after) - V(s_now)

    Select driver with maximum edge_weight.
    """

    def __init__(self,
                 v_network: VNetwork,
                 env,  # RideHailingEnv instance
                 alpha: float = 1.0,
                 gamma: float = 0.95,
                 pickup_norm_km: float = 1.5,
                 device: Optional[torch.device] = None):
        self.v_net = v_network
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.pickup_norm_km = pickup_norm_km
        self.device = device or next(v_network.parameters()).device

    def select_action(
        self,
        candidate_driver_ids: List[int],
        candidate_pickup_kms: List[float],
        passenger_dest_zone: int,
        passenger_trip_steps: Optional[int] = None,
    ) -> Tuple[int, np.ndarray, dict]:
        """
        Select driver from candidates using V-greedy.

        Args:
            candidate_driver_ids: list of real driver_ids (length K)
            candidate_pickup_kms: pickup distances in km (length K)
            passenger_dest_zone: zone of passenger destination (0-13)
            passenger_trip_steps: estimated trip duration in steps
                (for t_after). If None, defaults to 10 steps (~20 min,
                step_minutes=2). trip_duration in steps is not exposed
                by the ABM, so default is used (authorised for smoke).

        Returns:
            (best_idx, edge_weights, diagnostics)
            - best_idx: index in candidate list (0..K-1) of selected driver
            - edge_weights: array of edge weights per candidate
            - diagnostics: dict with intermediate values for debugging
        """
        K = len(candidate_driver_ids)
        if K == 0:
            return -1, np.array([]), {}

        current_step = int(self.env._model.schedule.steps)

        if passenger_trip_steps is None:
            passenger_trip_steps = 10  # ~20 min (step_minutes=2)

        s_now_list = []
        s_after_list = []

        for i, did in enumerate(candidate_driver_ids):
            # s_now: راننده در ناحیه‌ی جاری و زمان جاری
            s_now = self.env.get_driver_state(driver_id=did)
            s_now_list.append(s_now)

            # s_after: راننده در مقصد مسافر، پس از اتمام سفر
            # t_after = جاری + گام‌های pickup + گام‌های سفر
            # تخمین pickup: pickup_km / 1.5 km در هر گام
            pickup_steps = int(np.ceil(candidate_pickup_kms[i] / 1.5))
            t_after = current_step + pickup_steps + passenger_trip_steps

            s_after = self.env.get_driver_state(
                driver_id=did,
                override_zone=passenger_dest_zone,
                override_time_step=t_after,
            )
            s_after_list.append(s_after)

        s_now_batch = np.stack(s_now_list).astype(np.float32)      # (K,40)
        s_after_batch = np.stack(s_after_list).astype(np.float32)  # (K,40)

        v_now = np.atleast_1d(self.v_net.predict(s_now_batch))     # (K,)
        v_after = np.atleast_1d(self.v_net.predict(s_after_batch)) # (K,)

        pickups = np.array(candidate_pickup_kms, dtype=np.float32)
        r_immediate = 1.0 - self.alpha * (pickups / self.pickup_norm_km)

        edge_weights = r_immediate + self.gamma * v_after - v_now

        best_idx = int(np.argmax(edge_weights))

        diagnostics = {
            'r_immediate': r_immediate,
            'v_now': v_now,
            'v_after': v_after,
            'delta_v': v_after - v_now,
            'edge_weights': edge_weights,
            'best_idx': best_idx,
            'best_pickup_km': float(pickups[best_idx]),
        }

        return best_idx, edge_weights, diagnostics
