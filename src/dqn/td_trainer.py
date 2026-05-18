"""
TD(0) trainer for V-network in ride-hailing dispatch.

Reference: Sutton & Barto (2018) Ch.6 — TD(0) prediction.
Adapted for DiDi-style state-value learning (Tang et al. 2019).
"""

import copy
import time
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, List, Dict
from dataclasses import dataclass

from src.dqn.v_network import VNetwork
from src.dqn.v_greedy_policy import VGreedyPolicy

logger = logging.getLogger(__name__)


@dataclass
class Transition:
    """Single transition for V-learning."""
    s_now: np.ndarray       # (40,)
    reward: float           # scalar
    s_after: np.ndarray     # (40,)
    is_terminal: bool = False


class TDTrainer:
    """
    On-policy TD(0) trainer for V-network.

    Per-episode rollout, batched gradient updates every batch_size
    transitions, target network sync every target_sync_freq episodes.
    """

    def __init__(self,
                 env,
                 v_network: VNetwork,
                 policy: VGreedyPolicy,
                 lr: float = 1e-3,
                 gamma: float = 0.95,
                 batch_size: int = 32,
                 target_sync_freq: int = 5,
                 device: Optional[torch.device] = None,
                 grad_clip: float = 1.0):
        self.env = env
        self.v_online = v_network
        self.policy = policy

        self.device = device or next(v_network.parameters()).device

        self.v_target = copy.deepcopy(self.v_online).to(self.device)
        self.v_target.eval()
        for p in self.v_target.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.Adam(self.v_online.parameters(), lr=lr)

        self.gamma = gamma
        self.batch_size = batch_size
        self.target_sync_freq = target_sync_freq
        self.grad_clip = grad_clip

        self.rollout: List[Transition] = []

        self.episode_count = 0
        self.total_updates = 0
        self.total_transitions = 0
        self.steps_since_update: int = 0

    def train_episode(self, episode_idx: int,
                      seed: Optional[int] = None) -> Dict:
        """Train one episode: rollout, periodic updates, terminal update."""
        if seed is not None:
            obs, _ = self.env.reset(seed=seed)
        else:
            obs, _ = self.env.reset()

        self.rollout.clear()
        self.steps_since_update = 0  # reset برای هر episode

        ep_reward = 0.0
        ep_n_decisions = 0
        ep_losses: List[float] = []
        ep_v_now_mean: List[float] = []

        done = False
        t_start = time.time()

        while not done:
            cands = self.env._candidates
            n_cands = len(cands) if cands else 0

            s_now_arr = None
            s_after_arr = None
            diag = None

            if n_cands > 0:
                cand_ids = [int(c.driver.driver_id) for c in cands]
                cand_pickups = [float(c.pickup_dist_km) for c in cands]
                cur_passenger = cands[0].passenger
                p_dest_zone = int(cur_passenger.dest_zone)

                best_idx, edges, diag = self.policy.select_action(
                    candidate_driver_ids=cand_ids,
                    candidate_pickup_kms=cand_pickups,
                    passenger_dest_zone=p_dest_zone,
                )

                s_now_arr = self.policy.env.get_driver_state(
                    driver_id=cand_ids[best_idx]
                )

                current_step = int(self.env._model.schedule.steps)
                pickup_steps = int(np.ceil(cand_pickups[best_idx] / 1.5))
                t_after = current_step + pickup_steps + 10

                s_after_arr = self.policy.env.get_driver_state(
                    driver_id=cand_ids[best_idx],
                    override_zone=p_dest_zone,
                    override_time_step=t_after,
                )

                env_action = best_idx
                ep_n_decisions += 1
            else:
                env_action = self.env.K_MAX

            next_obs, reward, terminated, truncated, info = self.env.step(
                env_action
            )
            done = terminated or truncated

            if n_cands > 0 and s_now_arr is not None:
                self.rollout.append(Transition(
                    s_now=s_now_arr,
                    reward=float(reward),
                    s_after=s_after_arr,
                    is_terminal=done,
                ))
                if diag is not None and 'v_now' in diag:
                    ep_v_now_mean.append(float(np.mean(diag['v_now'])))

            ep_reward += float(reward)

            # آپدیت غیرهمپوشان: هر batch_size تصمیمِ واقعی، یک آپدیت
            if n_cands > 0 and s_now_arr is not None:
                self.steps_since_update += 1

                if self.steps_since_update >= self.batch_size:
                    loss = self._update_v(self.rollout[-self.batch_size:])
                    ep_losses.append(loss)
                    self.steps_since_update = 0

            obs = next_obs

        if len(self.rollout) > 0:
            remainder = len(self.rollout) % self.batch_size
            if remainder > 0:
                loss = self._update_v(self.rollout[-remainder:])
                ep_losses.append(loss)

        self.episode_count += 1
        if self.episode_count % self.target_sync_freq == 0:
            self.v_target.load_state_dict(self.v_online.state_dict())
            logger.debug("Target network synced at episode %d",
                         self.episode_count)

        summary = info.get('episode_summary', {}) if info else {}
        wall_time = time.time() - t_start

        return {
            'episode': episode_idx,
            'ep_reward': ep_reward,
            'avg_loss': float(np.mean(ep_losses)) if ep_losses else 0.0,
            'avg_v_now': (
                float(np.mean(ep_v_now_mean)) if ep_v_now_mean else 0.0
            ),
            'n_decisions': ep_n_decisions,
            'n_transitions': len(self.rollout),
            'n_updates': len(ep_losses),
            'wall_time': wall_time,
            # نکته: episode_summary کلیدهای mean_WT_min/mean_DU دارد،
            # نه WT/DU (انحراف env-specific — اصلاح‌شده).
            'CR': float(summary.get('CR', 0.0)),
            'WT': float(summary.get('mean_WT_min', 0.0)),
            'DU': float(summary.get('mean_DU', 0.0)),
        }

    def _update_v(self, batch: List[Transition]) -> float:
        """One gradient update on a batch of transitions."""
        states_now = torch.from_numpy(
            np.stack([t.s_now for t in batch])
        ).float().to(self.device)
        states_after = torch.from_numpy(
            np.stack([t.s_after for t in batch])
        ).float().to(self.device)
        rewards = torch.tensor(
            [t.reward for t in batch], dtype=torch.float32
        ).to(self.device)
        terminals = torch.tensor(
            [t.is_terminal for t in batch], dtype=torch.float32
        ).to(self.device)

        with torch.no_grad():
            v_target_after = self.v_target(states_after)
            td_target = rewards + self.gamma * v_target_after * (1 - terminals)

        v_pred = self.v_online(states_now)
        loss = nn.functional.mse_loss(v_pred, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.v_online.parameters(), self.grad_clip
        )
        self.optimizer.step()

        self.total_updates += 1
        self.total_transitions += len(batch)
        return float(loss.item())

    def save_checkpoint(self, path: str, episode: int):
        """Save model checkpoint."""
        torch.save({
            'v_online': self.v_online.state_dict(),
            'v_target': self.v_target.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'episode': episode,
            'total_updates': self.total_updates,
            'total_transitions': self.total_transitions,
        }, path)
        logger.info("Checkpoint saved: %s", path)

    def load_checkpoint(self, path: str) -> int:
        """Load checkpoint, return last episode."""
        ckpt = torch.load(path, map_location=self.device)
        self.v_online.load_state_dict(ckpt['v_online'])
        self.v_target.load_state_dict(ckpt['v_target'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.total_updates = ckpt.get('total_updates', 0)
        self.total_transitions = ckpt.get('total_transitions', 0)
        return ckpt.get('episode', 0)
