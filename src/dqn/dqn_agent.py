"""
عامل DQN با معماری Q(s, a_features) — مرحله ۷، فصل ۳.

نکات کلیدی:
    - سیاست no-op سختگیرانه (تصمیم تأییدشده): عامل همیشه با argmax یک
      کاندید معتبر را انتخاب می‌کند. اقدام K_MAX (no-op) فقط وقتی صادر
      می‌شود که هیچ کاندیدی نباشد (اجباری؛ خود gym_env هندل می‌کند).
    - معماری Q(s, a_features): شبکه به ازای هر کاندید یک‌بار صدا زده
      می‌شود. در update، state تکثیر (tile) و با هر a_features الحاق
      می‌شود، یک forward دسته‌ای زده می‌شود، سپس به ساختار per-sample
      بازگردانده می‌شود.
    - هر دو نسخه: DQN استاندارد (پیش‌فرض) و Double DQN (با پرچم).
        استاندارد: y = r + γ · max_a' Q_target(s', a')
        Double:    y = r + γ · Q_target(s', argmax_a' Q_online(s', a'))
    - به‌روزرسانی سخت (hard) شبکه‌ی هدف هر ۱۰۰۰ گام ABM (توسط trainer).
    - زیان Huber، بهینه‌ساز Adam (جدول ۳-۷).
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.dqn.network import QNetwork

_NEG_INF = -1e9


class DQNAgent:
    """عامل DQN با ε-greedy و معماری Q(s, a_features)."""

    def __init__(
        self,
        state_dim: int = 32,
        action_features_dim: int = 8,
        hidden_dims=(256, 256, 128),
        k_max: int = 50,
        lr: float = 5e-4,
        gamma: float = 0.95,
        double_dqn: bool = False,
        device: Optional[str] = None,
        seed: int = 0,
    ) -> None:
        self.state_dim = int(state_dim)
        self.action_features_dim = int(action_features_dim)
        self.k_max = int(k_max)
        self.gamma = float(gamma)
        self.double_dqn = bool(double_dqn)

        self.device = torch.device(
            device if device is not None else "cpu"
        )
        torch.manual_seed(seed)

        self.q_online = QNetwork(
            state_dim, action_features_dim, hidden_dims
        ).to(self.device)
        self.q_target = QNetwork(
            state_dim, action_features_dim, hidden_dims
        ).to(self.device)
        self.q_target.load_state_dict(self.q_online.state_dict())
        self.q_target.eval()

        self.optimizer = optim.Adam(self.q_online.parameters(), lr=float(lr))
        self.loss_fn = nn.SmoothL1Loss()  # Huber Loss
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # انتخاب اقدام (ε-greedy، no-op سختگیرانه)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_action(self, obs: Dict[str, np.ndarray], epsilon: float) -> int:
        """
        انتخاب اقدام طبق ε-greedy روی کاندیدهای معتبر.

        - اگر هیچ کاندیدی نباشد → K_MAX (no-op اجباری).
        - با احتمال ε → یک کاندید معتبر تصادفی.
        - با احتمال ۱−ε → argmax Q(s, a_features) روی کاندیدهای معتبر.
        """
        n = int(obs["n_candidates"])
        if n <= 0:
            return self.k_max  # no-op اجباری (هیچ کاندیدی نیست)

        if self._rng.random() < float(epsilon):
            return int(self._rng.integers(0, n))

        state = np.asarray(obs["state"], dtype=np.float32)            # (32,)
        cand = np.asarray(obs["candidates"], dtype=np.float32)[:n]    # (n, 8)

        state_rep = np.tile(state, (n, 1))                            # (n, 32)
        sa = np.concatenate([state_rep, cand], axis=1)                # (n, 40)
        sa_t = torch.from_numpy(sa).to(self.device)
        q = self.q_online(sa_t).squeeze(-1)                           # (n,)
        return int(torch.argmax(q).item())

    # ------------------------------------------------------------------
    # محاسبه‌ی max_a' Q(s', a') با ماسک کاندیدهای معتبر
    # ------------------------------------------------------------------

    def _next_state_value(
        self,
        next_states: torch.Tensor,      # (B, 32)
        next_cand_feats: torch.Tensor,  # (B, K, 8)
        next_masks: torch.Tensor,       # (B, K) ∈ {0,1}
    ) -> torch.Tensor:
        """
        مقدار حالت بعدی = max روی کاندیدهای معتبر.

        DQN استاندارد: max_a' Q_target(s', a')
        Double DQN:    Q_target(s', argmax_a' Q_online(s', a'))
        اگر هیچ کاندید معتبری نباشد → مقدار ۰.
        """
        b, k, _ = next_cand_feats.shape

        # تکثیر state و الحاق با ویژگی هر کاندید → (B, K, 40)
        s_rep = next_states.unsqueeze(1).expand(b, k, self.state_dim)
        sa = torch.cat([s_rep, next_cand_feats], dim=2)
        sa_flat = sa.reshape(b * k, self.state_dim + self.action_features_dim)

        mask = next_masks.bool()                       # (B, K)
        valid_any = mask.any(dim=1)                    # (B,)

        q_target_flat = self.q_target(sa_flat).reshape(b, k)  # (B, K)

        if self.double_dqn:
            q_online_flat = self.q_online(sa_flat).reshape(b, k)
            q_online_masked = q_online_flat.masked_fill(~mask, _NEG_INF)
            best_a = torch.argmax(q_online_masked, dim=1, keepdim=True)  # (B,1)
            next_v = q_target_flat.gather(1, best_a).squeeze(1)          # (B,)
        else:
            q_target_masked = q_target_flat.masked_fill(~mask, _NEG_INF)
            next_v = q_target_masked.max(dim=1).values                  # (B,)

        # نمونه‌هایی که هیچ کاندید معتبری ندارند → مقدار ۰
        next_v = torch.where(
            valid_any, next_v, torch.zeros_like(next_v)
        )
        return next_v

    # ------------------------------------------------------------------
    # به‌روزرسانی یک گام آموزش
    # ------------------------------------------------------------------

    def update(self, batch: Dict[str, np.ndarray]) -> float:
        """
        یک گام آموزش روی یک دسته‌ی نمونه‌گیری‌شده. خروجی: مقدار loss.
        """
        dev = self.device
        states = torch.from_numpy(batch["states"]).to(dev)            # (B,32)
        action_feats = torch.from_numpy(batch["action_feats"]).to(dev)  # (B,8)
        rewards = torch.from_numpy(batch["rewards"]).to(dev)          # (B,)
        next_states = torch.from_numpy(batch["next_states"]).to(dev)  # (B,32)
        next_cf = torch.from_numpy(batch["next_cand_feats"]).to(dev)  # (B,K,8)
        next_mask = torch.from_numpy(batch["next_masks"]).to(dev)     # (B,K)
        dones = torch.from_numpy(batch["dones"]).to(dev)              # (B,)

        # Q فعلی برای جفت انتخاب‌شده
        sa = torch.cat([states, action_feats], dim=1)                # (B,40)
        q_pred = self.q_online(sa).squeeze(-1)                        # (B,)

        # هدف Bellman
        with torch.no_grad():
            next_v = self._next_state_value(next_states, next_cf, next_mask)
            y = rewards + self.gamma * (1.0 - dones) * next_v

        loss = self.loss_fn(q_pred, y)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_online.parameters(), max_norm=10.0)
        self.optimizer.step()

        return float(loss.item())

    # ------------------------------------------------------------------
    # شبکه‌ی هدف و ذخیره/بارگذاری
    # ------------------------------------------------------------------

    def hard_update_target(self) -> None:
        """کپی کامل وزن‌های online به target (هر ۱۰۰۰ گام ABM)."""
        self.q_target.load_state_dict(self.q_online.state_dict())

    @torch.no_grad()
    def mean_q_value(self, obs: Dict[str, np.ndarray]) -> float:
        """میانگین Q روی کاندیدهای معتبر (برای logging)."""
        n = int(obs["n_candidates"])
        if n <= 0:
            return 0.0
        state = np.asarray(obs["state"], dtype=np.float32)
        cand = np.asarray(obs["candidates"], dtype=np.float32)[:n]
        sa = np.concatenate([np.tile(state, (n, 1)), cand], axis=1)
        sa_t = torch.from_numpy(sa).to(self.device)
        return float(self.q_online(sa_t).mean().item())

    def save(self, path: str) -> None:
        """ذخیره‌ی state_dict شبکه‌ی online + متادیتا."""
        torch.save(
            {
                "q_online": self.q_online.state_dict(),
                "q_target": self.q_target.state_dict(),
                "double_dqn": self.double_dqn,
                "gamma": self.gamma,
            },
            path,
        )

    def load(self, path: str, map_location: Optional[str] = None) -> None:
        """بارگذاری وزن‌ها از فایل ذخیره‌شده."""
        ckpt = torch.load(
            path, map_location=map_location or self.device
        )
        self.q_online.load_state_dict(ckpt["q_online"])
        self.q_target.load_state_dict(
            ckpt.get("q_target", ckpt["q_online"])
        )
