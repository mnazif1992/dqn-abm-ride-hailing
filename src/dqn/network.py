"""
شبکه عصبی Q(s, a_features) — مرحله ۷، بخش ۳-۶-۱ فصل ۳.

معماری ویژه‌ی این پایان‌نامه: به‌جای خروجی Top-K، شبکه برای هر جفت
کاندید یک اسکالر Q تولید می‌کند. ورودی = concat(state[32], a_features[8]) = 40.

ساختار (جدول معماری، بخش ۳-۶-۱):
    ورودی: ۴۰ نورون (d_in = d_s + d_a = ۳۲ + ۸)
    لایه پنهان ۱: ۲۵۶ نورون، ReLU
    لایه پنهان ۲: ۲۵۶ نورون، ReLU
    لایه پنهان ۳: ۱۲۸ نورون، ReLU
    خروجی: ۱ نورون (Q-value اسکالر)، بدون فعال‌سازی (خطی)
    ≈ ۱۲۸٬۰۰۰ پارامتر قابل آموزش

توجیه: استفاده از ۳ لایه پنهان به‌جای ۲ لایه استاندارد DQN
(Mnih et al., 2015) به دلیل ناهمگنی ورودی (ترکیب one-hot باینری
و پیوسته). لایه سوم (۲۵۶→۱۲۸) قابلیت یادگیری نگاشت‌های غیرخطی
بین ویژگی‌ها را فراهم می‌کند.
"""
from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    شبکه‌ی ارزش Q(s, a_features; θ).

    ورودی: تنسور (batch, state_dim + action_features_dim)
    خروجی: تنسور (batch, 1) — مقدار Q اسکالر برای هر جفت کاندید
    """

    def __init__(
        self,
        state_dim: int = 32,
        action_features_dim: int = 8,
        hidden_dims: Sequence[int] = (256, 256, 128),
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_features_dim = int(action_features_dim)
        self.input_dim = self.state_dim + self.action_features_dim

        layers: List[nn.Module] = []
        prev = self.input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, int(h)))
            layers.append(nn.ReLU())
            prev = int(h)
        layers.append(nn.Linear(prev, 1))  # خروجی خطی (بدون ReLU)

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        """مقداردهی اولیه‌ی Kaiming برای لایه‌های ReLU."""
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, state_action_concat: torch.Tensor) -> torch.Tensor:
        """
        محاسبه‌ی Q برای دسته‌ای از جفت‌های (state, a_features).

        ورودی:
            state_action_concat: (batch, input_dim) — concat حالت و ویژگی اقدام

        خروجی:
            (batch, 1) — مقدار Q
        """
        return self.net(state_action_concat)
