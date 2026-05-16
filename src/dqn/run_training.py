"""
Orchestrator آموزش DQN (CLI) — مرحله ۷ فصل ۳.

اجرا:
    # test run کوتاه برای validation پایپ‌لاین
    python -m src.dqn.run_training --n-episodes 50

    # آموزش واقعی (standard DQN، الگوریتم اصلی فصل ۳)
    python -m src.dqn.run_training --n-episodes 500

    # نسخه‌ی بهبودیافته (Double DQN، برای مقایسه‌ی فصل ۴)
    python -m src.dqn.run_training --n-episodes 500 --double-dqn

    # انتخاب صریح device
    python -m src.dqn.run_training --device mps

device پیش‌فرض: تشخیص خودکار MPS (مک M-series) → سپس CUDA → سپس CPU.

خروجی‌ها:
    experiments/logs/dqn_training.csv      — لاگ هر اپیزود
    experiments/models/dqn_best.pt         — بهترین مدل (val_reward)
    experiments/models/dqn_final.pt        — مدل نهایی
    experiments/models/dqn_ep{N}.pt        — checkpointهای دوره‌ای
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _auto_device() -> str:
    """تشخیص خودکار بهترین device موجود (MPS → CUDA → CPU)."""
    import torch

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def parse_args() -> argparse.Namespace:
    """تجزیه‌ی آرگومان‌های خط فرمان."""
    p = argparse.ArgumentParser(description="Train DQN on RideHailingEnv.")
    p.add_argument(
        "--config", type=str,
        default="experiments/configs/abm_calibrated.yaml",
    )
    p.add_argument(
        "--targets", type=str, default="data/calibration/targets.json"
    )
    p.add_argument(
        "--zones", type=str, default="data/calibration/zones.json"
    )
    p.add_argument(
        "--n-episodes", type=int, default=None,
        help="تعداد اپیزود (پیش‌فرض: dqn.n_episodes از config)",
    )
    p.add_argument(
        "--device", type=str, default=None,
        choices=["cpu", "mps", "cuda"],
        help="device (پیش‌فرض: تشخیص خودکار)",
    )
    p.add_argument(
        "--double-dqn", action="store_true",
        help="استفاده از Double DQN (پیش‌فرض: DQN استاندارد)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--log-csv", type=str,
        default="experiments/logs/dqn_training.csv",
    )
    p.add_argument(
        "--models-dir", type=str, default="experiments/models",
    )
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    from src.abm.utils import (
        load_config, load_targets, load_zones, setup_logging,
    )
    setup_logging(level=logging.DEBUG if args.debug else logging.INFO)

    root = Path(__file__).resolve().parents[2]
    config = load_config(root / args.config)
    targets = load_targets(root / args.targets)
    zones_data = load_zones(root / args.zones)

    dqn_cfg = dict(config.get("dqn", {}))
    n_episodes = (
        int(args.n_episodes)
        if args.n_episodes is not None
        else int(dqn_cfg.get("n_episodes", 1000))
    )
    device = args.device or _auto_device()
    logger.info("Device: %s | n_episodes: %d | double_dqn: %s",
                device, n_episodes, args.double_dqn)

    from src.abm.gym_env import RideHailingEnv
    from src.dqn.dqn_agent import DQNAgent
    from src.dqn.replay_buffer import ExperienceReplay
    from src.dqn.trainer import DQNTrainer

    state_dim = int(dqn_cfg.get("state_dim", 32))
    a_feat_dim = int(dqn_cfg.get("action_features_dim", 8))
    hidden_dims = tuple(dqn_cfg.get("hidden_dims", [256, 256, 128]))
    k_max = int(RideHailingEnv.K_MAX)

    # بافر را اول می‌سازیم تا به env پاس داده شود (پاداش تکمیلِ retroactive، گزینه D)
    buffer = ExperienceReplay(
        capacity=int(dqn_cfg.get("buffer_size", 100_000)),
        state_dim=state_dim,
        action_features_dim=a_feat_dim,
        k_max=k_max,
        seed=args.seed,
    )

    # ساخت محیط با رفرنس buffer برای پاداش تکمیل
    env = RideHailingEnv(
        config=config,
        targets=targets,
        zones_data=zones_data,
        seed=args.seed,
        replay_buffer=buffer,
    )
    logger.info("Reward mode: %s | completion_bonus: %.2f",
                env.reward_mode, env._completion_bonus)

    agent = DQNAgent(
        state_dim=state_dim,
        action_features_dim=a_feat_dim,
        hidden_dims=hidden_dims,
        k_max=k_max,
        lr=float(dqn_cfg.get("learning_rate", 5e-4)),
        gamma=float(dqn_cfg.get("gamma", 0.95)),
        double_dqn=bool(args.double_dqn),
        device=device,
        seed=args.seed,
    )

    trainer = DQNTrainer(
        env=env,
        agent=agent,
        buffer=buffer,
        batch_size=int(dqn_cfg.get("batch_size", 64)),
        warmup_steps=int(dqn_cfg.get("warmup_steps", 5_000)),
        target_update_freq=int(dqn_cfg.get("target_update_freq", 1_000)),
        epsilon_start=float(dqn_cfg.get("epsilon_start", 1.0)),
        epsilon_end=float(dqn_cfg.get("epsilon_end", 0.05)),
        epsilon_decay_steps=int(dqn_cfg.get("epsilon_decay_steps", 200_000)),
        early_stopping_patience=int(
            dqn_cfg.get("early_stopping_patience", 50)
        ),
        log_csv=str(root / args.log_csv),
        models_dir=str(root / args.models_dir),
    )

    try:
        summary = trainer.train(n_episodes=n_episodes)
    finally:
        env.close()

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info("best_val_reward : %.4f", summary["best_val_reward"])
    logger.info("best_model      : %s", summary["best_model"])
    logger.info("final_model     : %s", summary["final_model"])
    logger.info("stopped_early   : %s", summary["stopped_early"])
    logger.info("total_wall_s    : %.1f", summary["total_wall_s"])
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
