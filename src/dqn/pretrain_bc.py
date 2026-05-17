"""
Behavioral Cloning Pretraining — مرحله ۳ از v3 (Imitation Learning + RL).

داده‌ی expert را از expert_data.npz بارگذاری می‌کند و یک QNetwork را
با cross-entropy loss آموزش می‌دهد تا expert_action را پیش‌بینی کند.

این مدل به‌عنوان warm-start برای RL fine-tuning (مرحله ۴) استفاده
می‌شود — شبکه پیش از RL یک policy منطقی (مشابه Greedy) را می‌داند،
به‌جای شروع از صفر (که در v1/v2 شکست خورد).

نکته‌ی فنی مهم: ~۹۰.۵٪ نمونه‌ها expert_action=0 هستند (به‌خاطر
مرتب‌سازی env بر اساس pickup_dist). برای جلوگیری از trivial policy:
  1) cross-entropy روی کاندیدهای ماسک‌شده (نه ۵۰ ثابت)
  2) گزارش جداگانه‌ی accuracy روی نمونه‌های غیر-صفر (تنوع واقعی)
  3) نظارت بر اینکه شبکه واقعاً features را یاد می‌گیرد نه میانبر «همیشه ۰»
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class ExpertDataset(Dataset):
    """Dataset wrapper برای داده‌ی expert ذخیره‌شده در .npz."""

    def __init__(self, npz_path: str) -> None:
        data = np.load(npz_path)
        self.states = torch.from_numpy(data["states"]).float()          # (N,32)
        self.candidates = torch.from_numpy(data["candidates"]).float()  # (N,50,8)
        self.masks = torch.from_numpy(data["masks"]).bool()             # (N,50)
        self.expert_actions = torch.from_numpy(
            data["expert_actions"]
        ).long()                                                        # (N,)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> dict:
        return {
            "state": self.states[idx],
            "candidates": self.candidates[idx],
            "mask": self.masks[idx],
            "expert_action": self.expert_actions[idx],
        }


def compute_logits(
    network: nn.Module,
    state: torch.Tensor,
    candidates: torch.Tensor,
) -> torch.Tensor:
    """
    محاسبه‌ی logits برای هر کاندید با معماری Q(s, a_features).

    ورودی:
        network: QNetwork
        state: (B, 32)
        candidates: (B, K, 8)

    خروجی:
        logits: (B, K)
    """
    b, k, _ = candidates.shape
    state_rep = state.unsqueeze(1).expand(-1, k, -1)       # (B, K, 32)
    sa = torch.cat([state_rep, candidates], dim=-1)        # (B, K, 40)
    sa_flat = sa.reshape(b * k, -1)                        # (B*K, 40)
    logits = network(sa_flat).reshape(b, k)               # (B, K)
    return logits


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Behavioral Cloning pretraining for v3 DQN warm-start."
    )
    p.add_argument(
        "--expert-data", type=str,
        default="experiments/expert_data/expert_data.npz",
    )
    p.add_argument(
        "--output-model", type=str,
        default="experiments/models/dqn_v3_pretrained.pt",
    )
    p.add_argument(
        "--config", type=str,
        default="experiments/configs/abm_calibrated.yaml",
    )
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--device", type=str, default=None,
        choices=["cpu", "mps", "cuda"],
    )
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Device (auto-detect: MPS → CUDA → CPU)
    if args.device is None:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # بارگذاری داده
    print(f"Loading expert data from {args.expert_data}...")
    dataset = ExpertDataset(args.expert_data)
    print(f"Loaded {len(dataset)} samples")

    # تقسیم train/val
    n_total = len(dataset)
    n_val = int(args.val_split * n_total)
    n_train = n_total - n_val
    torch.manual_seed(args.seed)
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val]
    )
    print(f"Train: {n_train}, Val: {n_val}")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # شبکه (هم‌ابعاد با dqn_agent / network.py)
    from src.dqn.network import QNetwork

    network = QNetwork(
        state_dim=32, action_features_dim=8, hidden_dims=(256, 256, 128)
    ).to(device)
    optimizer = optim.Adam(network.parameters(), lr=args.lr)

    print("\n=== Training ===")
    train_acc = val_acc = val_acc_nz = 0.0
    for epoch in range(args.n_epochs):
        # ---------- Train ----------
        network.train()
        train_loss = 0.0
        train_correct = train_total = 0
        train_correct_nz = train_total_nz = 0

        for batch in train_loader:
            state = batch["state"].to(device)
            candidates = batch["candidates"].to(device)
            mask = batch["mask"].to(device)
            expert_action = batch["expert_action"].to(device)

            logits = compute_logits(network, state, candidates)   # (B,K)
            logits_masked = logits.masked_fill(~mask, -1e9)
            loss = nn.functional.cross_entropy(logits_masked, expert_action)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * state.size(0)
            preds = logits_masked.argmax(dim=1)
            train_correct += (preds == expert_action).sum().item()
            train_total += state.size(0)

            nz = expert_action != 0
            if nz.sum() > 0:
                train_correct_nz += (
                    (preds == expert_action) & nz
                ).sum().item()
                train_total_nz += int(nz.sum().item())

        train_loss /= max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)
        train_acc_nz = (
            train_correct_nz / train_total_nz if train_total_nz > 0 else 0.0
        )

        # ---------- Val ----------
        network.eval()
        val_loss = 0.0
        val_correct = val_total = 0
        val_correct_nz = val_total_nz = 0

        with torch.no_grad():
            for batch in val_loader:
                state = batch["state"].to(device)
                candidates = batch["candidates"].to(device)
                mask = batch["mask"].to(device)
                expert_action = batch["expert_action"].to(device)

                logits = compute_logits(network, state, candidates)
                logits_masked = logits.masked_fill(~mask, -1e9)
                loss = nn.functional.cross_entropy(
                    logits_masked, expert_action
                )

                val_loss += loss.item() * state.size(0)
                preds = logits_masked.argmax(dim=1)
                val_correct += (preds == expert_action).sum().item()
                val_total += state.size(0)

                nz = expert_action != 0
                if nz.sum() > 0:
                    val_correct_nz += (
                        (preds == expert_action) & nz
                    ).sum().item()
                    val_total_nz += int(nz.sum().item())

        val_loss /= max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)
        val_acc_nz = (
            val_correct_nz / val_total_nz if val_total_nz > 0 else 0.0
        )

        print(
            f"Epoch {epoch+1}/{args.n_epochs} | "
            f"Train: loss={train_loss:.4f} acc={train_acc:.3f} "
            f"acc_nz={train_acc_nz:.3f} | "
            f"Val: loss={val_loss:.4f} acc={val_acc:.3f} "
            f"acc_nz={val_acc_nz:.3f}"
        )

    # ذخیره (سازگار با DQNAgent.load: کلیدهای q_online/q_target)
    output_path = Path(args.output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "q_online": network.state_dict(),
            "q_target": network.state_dict(),  # یکسان برای شروع RL
            "double_dqn": True,
            "gamma": 0.95,
            "pretrained_with_bc": True,
            "n_epochs": args.n_epochs,
            "final_train_acc": train_acc,
            "final_val_acc": val_acc,
            "final_val_acc_nonzero": val_acc_nz,
        },
        output_path,
    )

    print(f"\n✅ Saved pretrained model to {output_path}")
    print(
        f"Final: train_acc={train_acc:.3f}, val_acc={val_acc:.3f}, "
        f"val_acc_nonzero={val_acc_nz:.3f}"
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
