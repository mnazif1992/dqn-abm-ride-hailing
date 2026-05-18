"""
V-network for state-value estimation in DiDi-style ride-hailing dispatch.

Reference: Tang et al. 2019 (KDD), Qin et al. 2020 (INFORMS).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Union


class VNetwork(nn.Module):
    """
    State value network V(s) for ride-hailing dispatch.

    Input: driver state vector (40-dim from RideHailingEnv.get_driver_state)
    Output: scalar value V(s) representing expected future return

    Architecture:
        40 -> 128 -> 64 -> 1
        ReLU between hidden layers, linear output
    """

    DEFAULT_STATE_DIM = 40

    def __init__(self,
                 state_dim: int = 40,
                 hidden_dims: tuple = (128, 64)):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dims = hidden_dims

        layers = []
        prev_dim = state_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)

        # وزن‌دهی Xavier برای شروع پایدار
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (B, state_dim) or (state_dim,) tensor
        Returns:
            (B,) or scalar tensor — V(s) values
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
            v = self.net(state).squeeze(-1)
            return v.squeeze(0)
        return self.net(state).squeeze(-1)

    @torch.no_grad()
    def predict(self, state: Union[np.ndarray, torch.Tensor],
                device: torch.device = None) -> Union[float, np.ndarray]:
        """
        Convenience method for inference (during Hungarian matching).

        Args:
            state: numpy array or tensor — (state_dim,) or (N, state_dim)
            device: target device (default: same as model)
        Returns:
            float (single state) or numpy array (batch)
        """
        if device is None:
            device = next(self.parameters()).device

        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        state = state.to(device)

        single = state.dim() == 1
        out = self.forward(state)
        out_np = out.cpu().numpy()
        return float(out_np) if single else out_np
