"""
Run V-network training for ride-hailing dispatch.

Usage:
    python -m src.dqn.run_v_training --n-episodes 200
    python -m src.dqn.run_v_training --n-episodes 1000 --eval-every 20
"""

import argparse
import csv
import logging
import sys
from pathlib import Path
import numpy as np
import torch

from src.dqn.v_network import VNetwork
from src.dqn.v_greedy_policy import VGreedyPolicy
from src.dqn.td_trainer import TDTrainer
from src.abm.gym_env import RideHailingEnv
from src.abm.utils import load_config, load_targets, load_zones


def setup_logging(log_path: str):
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ]
    )


def evaluate(env, v_net, policy,
             val_seeds=(10000, 10001, 10002, 10003, 10004)):
    """Run policy on validation seeds, return mean stats."""
    v_net.eval()
    crs, wts, dus, rewards = [], [], [], []

    for seed in val_seeds:
        obs, _ = env.reset(seed=seed)
        ep_reward = 0.0
        done = False

        while not done:
            cands = env._candidates
            n_cands = len(cands) if cands else 0

            if n_cands > 0:
                cand_ids = [int(c.driver.driver_id) for c in cands]
                cand_pickups = [float(c.pickup_dist_km) for c in cands]
                p_dest_zone = int(cands[0].passenger.dest_zone)

                best_idx, _, _ = policy.select_action(
                    candidate_driver_ids=cand_ids,
                    candidate_pickup_kms=cand_pickups,
                    passenger_dest_zone=p_dest_zone,
                )
                env_action = best_idx
            else:
                env_action = env.K_MAX

            obs, reward, term, trunc, info = env.step(env_action)
            ep_reward += float(reward)
            done = term or trunc

        summary = info.get('episode_summary', {}) if info else {}
        crs.append(float(summary.get('CR', 0.0)))
        wts.append(float(summary.get('mean_WT_min', 0.0)))
        dus.append(float(summary.get('mean_DU', 0.0)))
        rewards.append(ep_reward)

    v_net.train()
    return {
        'val_reward_mean': float(np.mean(rewards)),
        'val_reward_std': float(np.std(rewards)),
        'val_CR': float(np.mean(crs)),
        'val_WT': float(np.mean(wts)),
        'val_DU': float(np.mean(dus)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', type=int, default=200)
    parser.add_argument('--eval-every', type=int, default=10)
    parser.add_argument('--save-every', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--target-sync-freq', type=int, default=5)
    parser.add_argument('--log-dir', type=str, default='experiments/logs')
    parser.add_argument('--model-dir', type=str,
                        default='experiments/models')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed-offset', type=int, default=0)
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device(
            'mps' if torch.backends.mps.is_available() else 'cpu'
        )
    else:
        device = torch.device(args.device)

    log_dir = Path(args.log_dir)
    model_dir = Path(args.model_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    csv_path = log_dir / 'v_training.csv'
    log_path = log_dir / 'v_training_full.log'

    setup_logging(str(log_path))
    logger = logging.getLogger(__name__)
    logger.info("=== V-network training started ===")
    logger.info("Args: %s", vars(args))
    logger.info("Device: %s", device)

    config = load_config('experiments/configs/abm_calibrated.yaml')
    targets = load_targets('data/calibration/targets.json')
    zones = load_zones('data/calibration/zones.json')

    env = RideHailingEnv(config=config, targets=targets,
                         zones_data=zones, seed=42)

    v_net = VNetwork(state_dim=40, hidden_dims=(128, 64)).to(device)
    policy = VGreedyPolicy(v_net, env, alpha=args.alpha,
                           gamma=args.gamma, device=device)
    trainer = TDTrainer(
        env=env, v_network=v_net, policy=policy,
        lr=args.lr, gamma=args.gamma,
        batch_size=args.batch_size,
        target_sync_freq=args.target_sync_freq,
        device=device,
    )

    csv_columns = [
        'episode', 'ep_reward', 'avg_loss', 'avg_v_now',
        'n_decisions', 'n_transitions', 'n_updates', 'wall_time_s',
        'train_CR', 'train_WT', 'train_DU',
        'val_reward_mean', 'val_reward_std', 'val_CR', 'val_WT', 'val_DU',
    ]
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(csv_columns)

    best_val_cr = 0.0

    for ep in range(args.n_episodes):
        seed = args.seed_offset + ep

        result = trainer.train_episode(episode_idx=ep, seed=seed)

        logger.info(
            "ep=%d reward=%.1f loss=%.4f v=%.3f CR_train=%.3f wall=%.1fs",
            ep, result['ep_reward'], result['avg_loss'],
            result['avg_v_now'], result['CR'], result['wall_time'],
        )

        eval_result = {}
        if (ep + 1) % args.eval_every == 0 or ep == 0:
            eval_result = evaluate(env, v_net, policy)
            logger.info(
                "  EVAL ep=%d: val_CR=%.4f val_WT=%.3f val_DU=%.3f",
                ep, eval_result['val_CR'], eval_result['val_WT'],
                eval_result['val_DU'],
            )
            if eval_result['val_CR'] > best_val_cr:
                best_val_cr = eval_result['val_CR']
                trainer.save_checkpoint(
                    str(model_dir / 'v_best.pt'), episode=ep
                )
                logger.info("  * New best val_CR: %.4f", best_val_cr)

        row = [
            ep, result['ep_reward'], result['avg_loss'],
            result['avg_v_now'], result['n_decisions'],
            result['n_transitions'], result['n_updates'],
            result['wall_time'], result['CR'], result['WT'],
            result['DU'],
            eval_result.get('val_reward_mean', ''),
            eval_result.get('val_reward_std', ''),
            eval_result.get('val_CR', ''),
            eval_result.get('val_WT', ''),
            eval_result.get('val_DU', ''),
        ]
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow(row)

        if (ep + 1) % args.save_every == 0:
            trainer.save_checkpoint(
                str(model_dir / f'v_ep{ep+1}.pt'), episode=ep
            )

    trainer.save_checkpoint(
        str(model_dir / 'v_final.pt'), episode=args.n_episodes - 1
    )
    logger.info("=== Training complete. Best val_CR: %.4f ===",
                best_val_cr)

    env.close()


if __name__ == '__main__':
    main()
