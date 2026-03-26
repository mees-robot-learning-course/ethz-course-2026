"""Training script for SO-100 action-chunking imitation learning.

Imports a model from hw3.model and trains it on
state -> action-chunk prediction using the processed zarr dataset.

Usage:
    python scripts/train.py --zarr datasets/processed/single_cube/processed_ee_xyz.zarr \
        --state-keys ... \
        --action-keys ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import zarr as zarr_lib
from hw3.dataset import (
    Normalizer,
    SO100ChunkDataset,
    load_and_merge_zarrs,
    load_zarr,
)
from hw3.model import BasePolicy, build_policy

# TODO: Any imports you want from torch or other libraries we use. Not allowed: libraries we don't use
from torch.utils.data import DataLoader, random_split

# TODO: Choose your own hyperparameters!
EPOCHS = 100 
BATCH_SIZE = 64
LR = 1e-3
VAL_SPLIT = 0.1
VALID_BACKBONES = ("mlp",)


def train_one_epoch(
    model: BasePolicy,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        states, action_chunks = batch
        # TODO: Implement the training step for one batch here.
        # This mostly: Get states and action_chunks onto the correct device, compute the loss, and step the optimizer.
        # Move data to device
        states = states.to(device)
        action_chunks = action_chunks.to(device)

        # Compute loss
        loss = model.compute_loss(states, action_chunks)
        
        # Compute gradients and step optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update total loss and number of batches
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: BasePolicy,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        states, action_chunks = batch
        # TODO: Implement the evaluation step for one batch here.
        # Move data to device
        states = states.to(device)
        action_chunks = action_chunks.to(device)

        # Compute loss
        loss = model.compute_loss(states, action_chunks)

        # Update total loss and number of batches
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def main() -> None:
    # TODO: You may add any cli arguments that make life easier for you like learning rate etc.
    parser = argparse.ArgumentParser(description="Train action-chunking policy.")
    parser.add_argument(
        "--task-config",
        type=Path,
        default=None,
        help=(
            "Path to a JSON task config file. If provided, config values override CLI values "
            "(for supported fields)."
        ),
    )
    parser.add_argument(
        "--zarr", type=Path, default=None, help="Path to processed .zarr store."
    )
    parser.add_argument(
        "--extra-zarr",
        type=Path,
        nargs="*",
        default=None,
        help="Optional additional processed .zarr stores to merge in (e.g. for DAgger).",
    )
    parser.add_argument(
        "--policy",
        choices=["obstacle", "multitask"],
        default="obstacle",
        help="Policy type: 'obstacle' for single-cube obstacle scene, 'multitask' for multicube (default: obstacle).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=16,
        help="Action chunk horizon H (default: 16).",
    )
    parser.add_argument(
        "--state-keys",
        nargs="+",
        default=None,
        help='State array key specs to concatenate, e.g. state_ee_xyz state_gripper "state_cube[:3]". '
        "Supports column slicing with [:N], [M:], [M:N]. "
        "If omitted, uses the state_key attribute from the zarr metadata.",
    )
    parser.add_argument(
        "--action-keys",
        nargs="+",
        default=None,
        help="Action array key specs to concatenate, e.g. action_ee_xyz action_gripper. "
        "Supports column slicing with [:N], [M:], [M:N]. "
        "If omitted, uses the action_key attribute from the zarr metadata.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help=f"Learning rate (default: module LR={LR}).",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        help=(
            f"Model backbone to use. Supported: {VALID_BACKBONES}. "
            "If provided in --task-config, it overrides CLI; defaults to 'mlp'."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    # ── load config (optional) ─────────────────────────────────────────
    if args.task_config is not None:
        cfg_path = args.task_config
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if not isinstance(cfg, dict):
            raise TypeError(f"Config must be a JSON object (dict), got: {type(cfg)}")

        # Config-first: if a field is present in config, it overrides CLI.
        if "zarr" in cfg and cfg["zarr"] is not None:
            args.zarr = Path(cfg["zarr"])
        if "extra_zarr" in cfg and cfg["extra_zarr"] is not None:
            args.extra_zarr = [Path(p) for p in cfg["extra_zarr"]]
        if "policy" in cfg and cfg["policy"] is not None:
            args.policy = str(cfg["policy"])
        if "chunk_size" in cfg and cfg["chunk_size"] is not None:
            args.chunk_size = int(cfg["chunk_size"])
        if "state_keys" in cfg and cfg["state_keys"] is not None:
            args.state_keys = cfg["state_keys"]
        if "action_keys" in cfg and cfg["action_keys"] is not None:
            args.action_keys = cfg["action_keys"]
        if "lr" in cfg and cfg["lr"] is not None:
            args.lr = float(cfg["lr"])
        if "backbone" in cfg and cfg["backbone"] is not None:
            args.backbone = str(cfg["backbone"])
        if "seed" in cfg and cfg["seed"] is not None:
            args.seed = int(cfg["seed"])

    # Default backbone if neither CLI nor config specified one.
    if args.backbone is None:
        args.backbone = "mlp"
    if args.backbone not in VALID_BACKBONES:
        raise SystemExit(
            f"Invalid backbone '{args.backbone}'. Supported backbones: {VALID_BACKBONES}"
        )
    if args.zarr is None:
        raise SystemExit(
            "You must specify --zarr, either directly on the CLI or via --task-config "
            "(JSON field 'zarr')."
        )

    # Require explicit state/action specification via CLI or config.
    # (We intentionally do NOT fall back to the dataset's default single key here.)
    if args.state_keys is None or args.action_keys is None:
        raise SystemExit(
            "You must specify both --state-keys and --action-keys, either directly on the CLI "
            "or via --task-config (with JSON fields 'state_keys' and 'action_keys')."
        )

    # Print resolved run configuration before starting training.
    print("\nResolved training configuration:")
    print(f"  zarr={args.zarr}")
    print(
        f"  extra_zarr={[str(p) for p in args.extra_zarr] if args.extra_zarr else []}"
    )
    print(f"  policy={args.policy}")
    print(f"  chunk_size={args.chunk_size}")
    print(f"  state_keys={args.state_keys}")
    print(f"  action_keys={args.action_keys}")
    print(f"  lr={args.lr if args.lr is not None else LR}")
    print(f"  backbone={args.backbone}")
    print(f"  seed={args.seed}")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── load data ─────────────────────────────────────────────────────
    zarr_paths = [args.zarr]
    if args.extra_zarr:
        zarr_paths.extend(args.extra_zarr)

    if len(zarr_paths) == 1:
        states, actions, ep_ends = load_zarr(
            args.zarr,
            state_keys=args.state_keys,
            action_keys=args.action_keys,
        )
    else:
        print(f"Merging {len(zarr_paths)} zarr stores: {[str(p) for p in zarr_paths]}")
        states, actions, ep_ends = load_and_merge_zarrs(
            zarr_paths,
            state_keys=args.state_keys,
            action_keys=args.action_keys,
        )
    normalizer = Normalizer.from_data(states, actions)

    dataset = SO100ChunkDataset(
        states,
        actions,
        ep_ends,
        chunk_size=args.chunk_size,
        normalizer=normalizer,
    )
    print(f"Dataset: {len(dataset)} samples, chunk_size={args.chunk_size}")
    print(f"  state_dim={states.shape[1]}, action_dim={actions.shape[1]}")

    # ── train / val split ─────────────────────────────────────────────
    n_val = max(1, int(len(dataset) * VAL_SPLIT))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    # ── model ─────────────────────────────────────────────────────────
    model = build_policy(
        args.policy,
        state_dim=states.shape[1],
        action_dim=actions.shape[1],
        chunk_size=args.chunk_size,
        backbone=args.backbone,
        # TODO: build with your desired specifications
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # TODO: implement an optimizer and scheduler
    lr = args.lr if args.lr is not None else LR
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=lr * 0.01,  # anneal close to (but not all the way to) zero
    )

    # ── training loop ─────────────────────────────────────────────────
    best_val = float("inf")

    # Derive action space tag from action keys (e.g. "ee_xyz", "joints")
    action_space = "unknown"
    if args.action_keys:
        for k in args.action_keys:
            base = k.split("[")[0]  # strip column slices
            if base != "action_gripper":
                action_space = base.removeprefix("action_")
                break

    save_name = f"best_model_{action_space}_{args.policy}.pt"

    n_dagger_eps = 0
    for zp in zarr_paths:
        z = zarr_lib.open_group(str(zp), mode="r")
        n_dagger_eps += z.attrs.get("num_dagger_episodes", 0)
    if n_dagger_eps > 0:
        save_name = f"best_model_{action_space}_{args.policy}_dagger{n_dagger_eps}ep.pt"
    # Default: checkpoints/<task>/
    if "multi_cube" in str(args.zarr):
        ckpt_dir = Path("./checkpoints/multi_cube")
    else:
        ckpt_dir = Path("./checkpoints/single_cube")
    save_path = ckpt_dir / save_name
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        scheduler.step()

        tag = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "normalizer": {
                        "state_mean": normalizer.state_mean,
                        "state_std": normalizer.state_std,
                        "action_mean": normalizer.action_mean,
                        "action_std": normalizer.action_std,
                    },
                    "chunk_size": args.chunk_size,
                    "policy_type": args.policy,
                    "backbone": args.backbone,
                    "d_model": int(getattr(model, "d_model", 128)),
                    "depth": int(getattr(model, "depth", 2)),
                    "state_keys": args.state_keys,
                    "action_keys": args.action_keys,
                    "state_dim": int(states.shape[1]),
                    "action_dim": int(actions.shape[1]),
                    "val_loss": val_loss,
                },
                save_path,
            )
            tag = " ✓ saved"

        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"train {train_loss:.6f} | val {val_loss:.6f}{tag}"
        )

    print(f"\nBest val loss: {best_val:.6f}")
    print(f"Checkpoint: {save_path}")


if __name__ == "__main__":
    main()
