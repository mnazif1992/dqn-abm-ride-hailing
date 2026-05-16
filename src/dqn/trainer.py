"""
حلقه‌ی آموزش DQN+ABM — مرحله ۷، الگوریتم ۲ و بخش ۳-۶-۳ فصل ۳.

تصمیمات تأییدشده‌ی طراحی:
    ۱) no-op سختگیرانه: عامل همیشه کاندید argmax را برمی‌گزیند؛ no-op
       فقط هنگام نبود کاندید (gym_env خودکار جلو می‌رود) و چنین گذاری
       در بافر ذخیره نمی‌شود (مطابق الگوریتم ۲ که فقط انتخاب کاندید
       گذار تولید می‌کند).
    ۲) واحد شمارش گام = گام ABM:
         epsilon = max(eps_end, eps_start − (eps_start−eps_end)
                       · global_abm_step / epsilon_decay_steps)
         به‌روزرسانی target هر target_update_freq گام ABM.
    ۳) همگرایی (§۳-۶-۳): هر ۱۰ اپیزود ارزیابی روی seedهای validation
       با ε=۰ (میانگین reward روی ۵ اپیزود)؛ early stopping با
       patience=۵۰ اپیزود؛ بهترین مدل ذخیره می‌شود.

seedهای validation: [10000, 10001, 10002, 10003, 10004] (مجزا از
training که seed=شماره‌ی اپیزود است) تا data/seed leakage رخ ندهد.
"""
from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.dqn.dqn_agent import DQNAgent
from src.dqn.replay_buffer import ExperienceReplay

logger = logging.getLogger(__name__)

VALIDATION_SEEDS: List[int] = [10000, 10001, 10002, 10003, 10004]


class DQNTrainer:
    """ارکستریتور حلقه‌ی بسته‌ی آموزش DQN روی محیط RideHailingEnv."""

    def __init__(
        self,
        env: Any,
        agent: DQNAgent,
        buffer: ExperienceReplay,
        *,
        batch_size: int = 64,
        warmup_steps: int = 5_000,
        target_update_freq: int = 1_000,
        train_freq: int = 1,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 200_000,
        eval_every: int = 10,
        eval_episodes: int = 5,
        early_stopping_patience: int = 50,
        checkpoint_every: int = 50,
        models_dir: str = "experiments/models",
        log_csv: str = "experiments/logs/dqn_training.csv",
    ) -> None:
        self.env = env
        self.agent = agent
        self.buffer = buffer
        self.k_max = agent.k_max

        self.batch_size = int(batch_size)
        self.warmup_steps = int(warmup_steps)
        self.target_update_freq = int(target_update_freq)
        self.train_freq = int(train_freq)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay_steps = int(epsilon_decay_steps)
        self.eval_every = int(eval_every)
        self.eval_episodes = int(eval_episodes)
        self.early_stopping_patience = int(early_stopping_patience)
        self.checkpoint_every = int(checkpoint_every)

        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.log_csv = Path(log_csv)
        self.log_csv.parent.mkdir(parents=True, exist_ok=True)

        # شمارنده‌ی سراسری گام ABM (واحد ε و target update)
        self.global_abm_step: int = 0
        self._last_target_update: int = 0
        self._best_val_reward: float = -float("inf")
        self._patience_counter: int = 0
        self._csv_initialized: bool = False

    # ------------------------------------------------------------------
    # epsilon
    # ------------------------------------------------------------------

    def _epsilon(self) -> float:
        frac = min(1.0, self.global_abm_step / max(1, self.epsilon_decay_steps))
        return max(
            self.epsilon_end,
            self.epsilon_start - (self.epsilon_start - self.epsilon_end) * frac,
        )

    # ------------------------------------------------------------------
    # logging
    # ------------------------------------------------------------------

    def _log_row(self, row: Dict[str, Any]) -> None:
        write_header = not self._csv_initialized and not self.log_csv.exists()
        with self.log_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        self._csv_initialized = True

    # ------------------------------------------------------------------
    # یک اپیزود آموزش
    # ------------------------------------------------------------------

    def _run_train_episode(self, episode: int) -> Dict[str, Any]:
        obs, info = self.env.reset(seed=episode)
        prev_abm = int(info.get("abm_step", 0))

        ep_reward = 0.0
        ep_loss_sum = 0.0
        ep_loss_count = 0
        ep_q_sum = 0.0
        ep_q_count = 0
        n_transitions = 0
        done = False

        while not done:
            epsilon = self._epsilon()
            action = self.agent.select_action(obs, epsilon)

            # logging میانگین Q (پیش از گام)
            if int(obs["n_candidates"]) > 0:
                ep_q_sum += self.agent.mean_q_value(obs)
                ep_q_count += 1

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = bool(terminated or truncated)
            ep_reward += float(reward)

            # فقط گذارهای انتخاب کاندید ذخیره می‌شوند (نه no-op اجباری)
            is_real_decision = action < self.k_max and action < int(
                obs["n_candidates"]
            )
            if is_real_decision:
                self.buffer.push(
                    state=np.asarray(obs["state"], dtype=np.float32),
                    action_features=np.asarray(
                        obs["candidates"][action], dtype=np.float32
                    ),
                    reward=float(reward),
                    next_state=np.asarray(
                        next_obs["state"], dtype=np.float32
                    ),
                    next_candidates_features=np.asarray(
                        next_obs["candidates"], dtype=np.float32
                    ),
                    next_mask=np.asarray(
                        next_obs["candidate_mask"], dtype=np.int8
                    ),
                    done=done,
                )
                n_transitions += 1

            # پیشروی شمارنده‌ی گام ABM
            cur_abm = int(info.get("abm_step", prev_abm))
            self.global_abm_step += max(0, cur_abm - prev_abm)
            prev_abm = cur_abm

            # گام آموزش
            if (
                len(self.buffer) > self.warmup_steps
                and n_transitions % self.train_freq == 0
            ):
                sample = self.buffer.sample(self.batch_size)
                loss = self.agent.update(sample)
                ep_loss_sum += loss
                ep_loss_count += 1

            # به‌روزرسانی سخت target هر target_update_freq گام ABM
            if (
                self.global_abm_step - self._last_target_update
                >= self.target_update_freq
            ):
                self.agent.hard_update_target()
                self._last_target_update = self.global_abm_step

            obs = next_obs

        return {
            "episode": episode,
            "ep_reward": ep_reward,
            "avg_loss": ep_loss_sum / ep_loss_count if ep_loss_count else 0.0,
            "avg_q": ep_q_sum / ep_q_count if ep_q_count else 0.0,
            "epsilon": self._epsilon(),
            "n_transitions": n_transitions,
            "n_completion_bonus": int(info.get("n_completion_bonus", 0)),
            "global_abm_step": self.global_abm_step,
            "buffer_size": len(self.buffer),
        }

    # ------------------------------------------------------------------
    # ارزیابی (ε=۰)
    # ------------------------------------------------------------------

    def evaluate(
        self, seeds: Optional[List[int]] = None
    ) -> Dict[str, float]:
        """اجرای greedy روی seedهای validation و میانگین reward + KPI."""
        seeds = seeds or VALIDATION_SEEDS[: self.eval_episodes]
        rewards: List[float] = []
        summaries: List[Dict[str, Any]] = []

        for sd in seeds:
            obs, info = self.env.reset(seed=sd)
            done = False
            total = 0.0
            while not done:
                action = self.agent.select_action(obs, epsilon=0.0)
                obs, reward, terminated, truncated, info = self.env.step(action)
                total += float(reward)
                done = bool(terminated or truncated)
            rewards.append(total)
            summaries.append(info.get("episode_summary", {}) or {})

        def _avg(key: str) -> float:
            vals = [float(s.get(key, 0.0)) for s in summaries if key in s]
            return float(np.mean(vals)) if vals else 0.0

        return {
            "val_reward_mean": float(np.mean(rewards)),
            "val_reward_std": float(np.std(rewards)),
            "val_CR": _avg("CR"),
            "val_WT": _avg("mean_WT_min"),
            "val_DU": _avg("mean_DU"),
        }

    # ------------------------------------------------------------------
    # حلقه‌ی اصلی آموزش
    # ------------------------------------------------------------------

    def train(self, n_episodes: int = 1000) -> Dict[str, Any]:
        """
        حلقه‌ی اصلی الگوریتم ۲ با early stopping.

        خروجی: dict خلاصه شامل بهترین val_reward و مسیر بهترین مدل.
        """
        logger.info(
            "Starting DQN training: n_episodes=%d, double_dqn=%s, device=%s",
            n_episodes, self.agent.double_dqn, self.agent.device,
        )
        best_path = str(self.models_dir / "dqn_best.pt")
        t0 = time.time()
        stopped_early = False

        for episode in range(1, int(n_episodes) + 1):
            ep_stats = self._run_train_episode(episode)

            row: Dict[str, Any] = dict(ep_stats)
            row["wall_time_s"] = round(time.time() - t0, 1)

            # ارزیابی دوره‌ای
            if episode % self.eval_every == 0:
                val = self.evaluate()
                row.update(val)
                logger.info(
                    "ep=%d reward=%.3f loss=%.4f q=%.3f eps=%.3f | "
                    "val_reward=%.3f val_CR=%.3f val_WT=%.3f val_DU=%.3f",
                    episode, ep_stats["ep_reward"], ep_stats["avg_loss"],
                    ep_stats["avg_q"], ep_stats["epsilon"],
                    val["val_reward_mean"], val["val_CR"],
                    val["val_WT"], val["val_DU"],
                )
                if val["val_reward_mean"] > self._best_val_reward:
                    self._best_val_reward = val["val_reward_mean"]
                    self._patience_counter = 0
                    self.agent.save(best_path)
                    logger.info(
                        "  ↑ new best val_reward=%.3f → saved %s",
                        self._best_val_reward, best_path,
                    )
                else:
                    self._patience_counter += 1
                    if self._patience_counter >= self.early_stopping_patience:
                        logger.info(
                            "Early stopping at episode %d "
                            "(patience=%d, best=%.3f)",
                            episode, self.early_stopping_patience,
                            self._best_val_reward,
                        )
                        stopped_early = True
            else:
                logger.info(
                    "ep=%d reward=%.3f loss=%.4f q=%.3f eps=%.3f cbonus=%d",
                    episode, ep_stats["ep_reward"], ep_stats["avg_loss"],
                    ep_stats["avg_q"], ep_stats["epsilon"],
                    ep_stats["n_completion_bonus"],
                )

            self._log_row(row)

            # checkpoint دوره‌ای
            if episode % self.checkpoint_every == 0:
                ckpt = str(self.models_dir / f"dqn_ep{episode}.pt")
                self.agent.save(ckpt)
                logger.info("  checkpoint saved → %s", ckpt)

            if stopped_early:
                break

        # ذخیره‌ی مدل نهایی
        final_path = str(self.models_dir / "dqn_final.pt")
        self.agent.save(final_path)
        logger.info(
            "Training done. best_val_reward=%.3f | best=%s final=%s",
            self._best_val_reward, best_path, final_path,
        )
        return {
            "best_val_reward": self._best_val_reward,
            "best_model": best_path,
            "final_model": final_path,
            "stopped_early": stopped_early,
            "total_wall_s": round(time.time() - t0, 1),
        }
