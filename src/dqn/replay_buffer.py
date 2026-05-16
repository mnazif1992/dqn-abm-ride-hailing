"""
بافر تجربه (Experience Replay) — مرحله ۷، جدول ۳-۷ فصل ۳.

هر گذار طبق رابطه ۳-۷ ذخیره می‌شود:
    (s_t, a_features_chosen, r_step, s_{t+1}, done)

نکته‌ی معماری Q(s, a_features): برای محاسبه‌ی هدف Bellman نیاز است
max روی کاندیدهای حالت بعدی محاسبه شود. بنابراین علاوه بر s_{t+1}،
ماتریس ویژگی کاندیدهای حالت بعدی و ماسک معتبربودن آن‌ها نیز ذخیره
می‌شود تا در update بتوان max_a' Q(s', a') را محاسبه کرد.

پیاده‌سازی به‌صورت ring-buffer روی numpy برای کارایی حافظه.
مرجع: Lin (1992)، Mnih et al. (2015).
"""
from __future__ import annotations

from typing import Dict

import numpy as np


class ExperienceReplay:
    """بافر بازپخش تجربه با نمونه‌گیری تصادفی یکنواخت."""

    def __init__(
        self,
        capacity: int = 100_000,
        state_dim: int = 32,
        action_features_dim: int = 8,
        k_max: int = 50,
        seed: int = 0,
    ) -> None:
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.action_features_dim = int(action_features_dim)
        self.k_max = int(k_max)
        self._rng = np.random.default_rng(seed)

        self._states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self._action_feats = np.zeros(
            (self.capacity, self.action_features_dim), dtype=np.float32
        )
        self._rewards = np.zeros((self.capacity,), dtype=np.float32)
        self._next_states = np.zeros(
            (self.capacity, self.state_dim), dtype=np.float32
        )
        self._next_cand_feats = np.zeros(
            (self.capacity, self.k_max, self.action_features_dim),
            dtype=np.float32,
        )
        self._next_masks = np.zeros(
            (self.capacity, self.k_max), dtype=np.int8
        )
        self._dones = np.zeros((self.capacity,), dtype=np.float32)

        self._ptr = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def push(
        self,
        state: np.ndarray,
        action_features: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        next_candidates_features: np.ndarray,
        next_mask: np.ndarray,
        done: bool,
    ) -> None:
        """ذخیره‌ی یک گذار در بافر (overwrite حلقوی پس از پر شدن)."""
        i = self._ptr
        self._states[i] = state
        self._action_feats[i] = action_features
        self._rewards[i] = float(reward)
        self._next_states[i] = next_state
        self._next_cand_feats[i] = next_candidates_features
        self._next_masks[i] = next_mask
        self._dones[i] = 1.0 if done else 0.0

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """نمونه‌گیری تصادفی یکنواخت یک دسته به‌صورت dict از آرایه‌های numpy."""
        idx = self._rng.integers(0, self._size, size=int(batch_size))
        return {
            "states": self._states[idx],
            "action_feats": self._action_feats[idx],
            "rewards": self._rewards[idx],
            "next_states": self._next_states[idx],
            "next_cand_feats": self._next_cand_feats[idx],
            "next_masks": self._next_masks[idx],
            "dones": self._dones[idx],
        }
