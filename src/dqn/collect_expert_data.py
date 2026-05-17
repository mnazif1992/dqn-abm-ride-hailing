"""
Expert Data Collector — Behavioral Cloning warm-start (v3).

Expert policy: Greedy خالص (بدون perturbation).

پایه: RideHailingEnv کاندیدها را بر اساس pickup_dist صعودی مرتب
می‌کند (gym_env._collect_candidates → cands.sort(key=pickup_dist_km)),
پس سیاست Greedy «نزدیک‌ترین راننده» دقیقاً معادل index 0 در
obs["candidates"] است.

یادداشت v2 (حذف perturbation): نسخه‌ی اول این فایل ۱۰٪ ε-perturbation
داشت (action تصادفی برای تنوع). اما در Behavioral Cloning این
supervision متناقض ایجاد کرد: برای حالت‌های تقریباً یکسان، گاهی
label=0 و گاهی یک اندیس تصادفی → cross-entropy روی برچسب‌های
متناقض → شبکه نتوانست حتی سیگنال درستِ «۰ را انتخاب کن» را یاد
بگیرد (val_acc=0.884 < baseline تریویال 0.905، loss روی آنتروپیِ
نویز ~۱.۰ کف کرد). چون expert ما deterministic است (Greedy = همیشه
نزدیک‌ترین = index 0)، perturbationِ تصادفی فقط سیگنال را آلوده
می‌کرد. تنوع ورودی از ~۲۸۴k حالتِ متنوع تأمین می‌شود — نیازی به
perturbationِ برچسب نیست. بنابراین EPSILON_PERTURB=0.0 شد.

استدلال انتخاب Greedy به‌جای Hungarian-resolve:
  1) Hungarian-resolve مخربِ state است (decide_accept/assign_to/
     mark_assigned) و قابل peek بدون خراب‌کردن env نیست.
  2) دامنه‌اش کلِ pool (همه‌ی available drivers × waiting passengers)
     است، با ۵۰ کاندیدِ obs ناسازگار؛ برچسب per-candidate برای بخش
     زیادی از تصمیم‌ها تعریف‌نشده می‌شد.
  3) حلقه‌ی re-solve با feedbackِ stochastic از reject هدایت می‌شود
     و آفلاین قابل بازتولید نیست.
Greedy تنها expertِ قابلِ برچسب‌گذاریِ faithfulِ per-candidate در این
env است (و اختلاف performance با Hungarian-resolve ناچیز است:
CR 0.609 vs 0.624).

خروجی: یک فایل .npz با کلیدهای states, candidates, masks,
expert_actions — ورودیِ مرحله‌ی Behavioral Cloning (v3 گام بعد).

اجرا (بعداً، با تأیید):
    python -m src.dqn.collect_expert_data --n-episodes 50
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

EPSILON_PERTURB = 0.0  # حذف شد: برای BC با expert deterministic مضر بود
                       # (supervision متناقض → loss روی نویز کف می‌کرد)


def parse_args() -> argparse.Namespace:
    """تجزیه‌ی آرگومان‌های خط فرمان."""
    p = argparse.ArgumentParser(
        description="Collect Greedy expert data for Behavioral Cloning (v3)."
    )
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
    p.add_argument("--n-episodes", type=int, default=50)
    p.add_argument(
        "--output-path", type=str,
        default="experiments/expert_data/expert_data.npz",
    )
    p.add_argument("--seed", type=int, default=0)
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

    rng = np.random.default_rng(args.seed)

    # محیط با reward_mode='shaped' برای سازگاری با v3 (پاداش در
    # data collection استفاده نمی‌شود، ولی mode یکسان نگه داشته می‌شود).
    # replay_buffer=None ⇒ پاداش تکمیلِ retroactive غیرفعال (لازم نیست).
    from src.abm.gym_env import RideHailingEnv

    env = RideHailingEnv(
        config=config,
        targets=targets,
        zones_data=zones_data,
        seed=args.seed,
        reward_mode="shaped",
        replay_buffer=None,
    )
    k_max = int(env.K_MAX)

    states: list = []
    candidates_list: list = []
    masks: list = []
    expert_actions: list = []

    for episode in range(args.n_episodes):
        # seedهای جمع‌آوری مجزا از training (seed=ep) و
        # validation ([10000..10004]) برای جلوگیری از leakage.
        obs, info = env.reset(seed=episode + 100_000)
        done = False

        while not done:
            n_cands = int(obs["n_candidates"])

            if n_cands == 0:
                # هیچ کاندیدی نیست → no-op اجباری؛ ذخیره نمی‌کنیم
                action = k_max
            else:
                # Greedy = index 0؛ با احتمال ε یک action معتبرِ
                # تصادفی (۱..n_cands-1) برای تنوع داده
                if n_cands > 1 and rng.random() < EPSILON_PERTURB:
                    action = int(rng.integers(1, n_cands))
                else:
                    action = 0

                states.append(np.asarray(obs["state"], dtype=np.float32).copy())
                candidates_list.append(
                    np.asarray(obs["candidates"], dtype=np.float32).copy()
                )
                masks.append(
                    np.asarray(obs["candidate_mask"], dtype=np.int8).copy()
                )
                expert_actions.append(int(action))

            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

        if (episode + 1) % 5 == 0:
            logger.info(
                "Episode %d/%d, samples collected: %d",
                episode + 1, args.n_episodes, len(states),
            )

    env.close()

    out_path = root / args.output_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    states_arr = np.array(states, dtype=np.float32)
    candidates_arr = np.array(candidates_list, dtype=np.float32)
    masks_arr = np.array(masks, dtype=np.int8)
    expert_actions_arr = np.array(expert_actions, dtype=np.int64)

    np.savez_compressed(
        out_path,
        states=states_arr,
        candidates=candidates_arr,
        masks=masks_arr,
        expert_actions=expert_actions_arr,
    )

    logger.info("=" * 60)
    logger.info("Saved %d expert samples to %s", len(states), out_path)
    logger.info(
        "Shapes: states=%s candidates=%s masks=%s expert_actions=%s",
        states_arr.shape, candidates_arr.shape,
        masks_arr.shape, expert_actions_arr.shape,
    )

    # توزیع expert_actions (دیباگ: باید ~۹۰٪ روی action=0 باشد)
    if len(expert_actions) > 0:
        unique, counts = np.unique(expert_actions_arr, return_counts=True)
        order = np.argsort(-counts)[:10]
        logger.info("Expert action distribution (top 10):")
        for i in order:
            logger.info(
                "  action=%d: %d (%.1f%%)",
                int(unique[i]), int(counts[i]),
                100.0 * counts[i] / len(expert_actions),
            )
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
