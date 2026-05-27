import argparse
import csv
import os
import sys
from dataclasses import asdict, fields

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Subset

from config import Config
from dataset import ASVspoof2019Dataset, ASVspoof5Dataset, ATADDDataset, MLAADDataset, collate_fn
from features import compute_mel, compute_mask
from models import build_model
from utils import set_seed
from RawBoost import LnL_convolutive_noise, ISD_additive_noise, SSI_additive_noise


def apply_rawboost(waveform: torch.Tensor, sample_rate: int, prob: float = 0.7) -> torch.Tensor:
    """Randomly chain up to 3 RawBoost noise types, each applied with probability `prob`."""
    x = waveform.numpy().astype(np.float64)
    if np.random.rand() < prob:
        x = LnL_convolutive_noise(
            x, N_f=5, nBands=5, minF=20, maxF=8000,
            minBW=100, maxBW=1000, minCoeff=10, maxCoeff=100,
            minG=0, maxG=0, minBiasLinNonLin=5, maxBiasLinNonLin=20,
            fs=sample_rate,
        )
    if np.random.rand() < prob:
        x = ISD_additive_noise(x, P=10, g_sd=2)
    if np.random.rand() < prob:
        x = SSI_additive_noise(
            x, SNRmin=10, SNRmax=40, nBands=5, minF=20, maxF=8000,
            minBW=100, maxBW=1000, minCoeff=10, maxCoeff=100,
            minG=0, maxG=0, fs=sample_rate,
        )
    x = np.clip(x, -1.0, 1.0)
    return torch.from_numpy(x).float()


def parse_args():
    parser = argparse.ArgumentParser(description="Train spectrogram autoencoder")
    parser.add_argument("--train_protocol", required=True)
    parser.add_argument("--train_root", required=True)
    parser.add_argument("--dev_protocol", default=None)
    parser.add_argument("--dev_root", default=None)
    parser.add_argument("--encoder", default="cnn", choices=["cnn", "vit_tiny"])
    parser.add_argument("--train_dataset", default="asv19", choices=["asv19", "atadd"],
                        help="Dataset class to use for the primary train split")
    parser.add_argument("--dev_dataset", default="asv19", choices=["asv19", "atadd"],
                        help="Dataset class for dev when not using --val_frac")
    parser.add_argument("--val_frac", type=float, default=0.0,
                        help="If >0, split this fraction of bonafide train samples as val "
                             "instead of loading a separate dev set")
    # Optional extra training datasets (bonafide only)
    parser.add_argument("--asv5_train_root", default=None)
    parser.add_argument("--asv5_train_protocol", default=None)
    parser.add_argument("--mlaad_train_root", default=None)
    parser.add_argument("--mlaad_train_protocol", default=None)

    # Allow overriding any Config field
    cfg_fields = {f.name: f for f in fields(Config)}
    for name, field in cfg_fields.items():
        if name in ("train_protocol", "train_root", "dev_protocol", "dev_root", "encoder"):
            continue
        t = field.type if isinstance(field.type, type) else str
        parser.add_argument(f"--{name}", type=t, default=None)

    return parser.parse_args()


def build_config(args) -> Config:
    cfg = Config()
    for f in fields(Config):
        val = getattr(args, f.name, None)
        if val is not None:
            setattr(cfg, f.name, val)
    cfg.encoder = args.encoder
    return cfg


def train_one_epoch(model, loader, optimizer, scheduler, device, cfg):
    model.train()
    total_loss = 0.0

    for waveforms, labels, audio_names in loader:
        if cfg.rawboost_prob > 0:
            waveforms = torch.stack([
                apply_rawboost(w, cfg.sample_rate, cfg.rawboost_prob) for w in waveforms
            ])
        mels = torch.stack([compute_mel(w.to(device), cfg) for w in waveforms])
        # mels: [B, 80, 500]

        masks = torch.stack([
            compute_mask(m, cfg) for m in mels
        ])
        # masks: [B, 80, 500], binary

        masked_input = (mels * masks).unsqueeze(1)   # [B, 1, 80, 500]
        target       = mels.unsqueeze(1)              # [B, 1, 80, 500]

        recon, z = model(masked_input)
        loss = F.mse_loss(recon, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate(model, loader, device, cfg):
    model.eval()
    total_loss = 0.0

    for waveforms, labels, audio_names in loader:
        mels = torch.stack([compute_mel(w.to(device), cfg) for w in waveforms])
        masks = torch.stack([
            compute_mask(m, cfg) for m in mels
        ])
        masked_input = (mels * masks).unsqueeze(1)
        target       = mels.unsqueeze(1)
        recon, _ = model(masked_input)
        total_loss += F.mse_loss(recon, target).item()

    return total_loss / max(len(loader), 1)


def main():
    args = parse_args()
    cfg = build_config(args)
    set_seed(42)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.save_dir, exist_ok=True)

    print("=" * 60)
    print("CONFIG")
    print("=" * 60)
    for f in fields(cfg):
        print(f"  {f.name}: {getattr(cfg, f.name)}")
    print(f"  device (resolved): {device}")
    print("=" * 60, flush=True)

    kw = dict(target_sample_rate=cfg.sample_rate, max_duration_seconds=cfg.max_duration_seconds)

    _DATASET_CLS = {"asv19": ASVspoof2019Dataset, "atadd": ATADDDataset}
    PrimaryTrainCls = _DATASET_CLS[args.train_dataset]
    DevCls = _DATASET_CLS[args.dev_dataset]

    primary_ds = PrimaryTrainCls(args.train_root, args.train_protocol, subset="bonafide", **kw)

    if args.val_frac > 0:
        n = len(primary_ds)
        indices = np.random.RandomState(42).permutation(n).tolist()
        n_val = int(n * args.val_frac)
        train_primary = Subset(primary_ds, indices[n_val:])
        dev_ds = Subset(primary_ds, indices[:n_val])
        print(f"Train/val split (val_frac={args.val_frac}): "
              f"{len(train_primary)} train, {n_val} val from {len(primary_ds)} bonafide samples")
    else:
        if not args.dev_root or not args.dev_protocol:
            raise ValueError("--dev_root and --dev_protocol are required when --val_frac is not set")
        train_primary = primary_ds
        dev_ds = DevCls(args.dev_root, args.dev_protocol, subset="bonafide", **kw)

    train_datasets = [train_primary]
    if args.asv5_train_root and args.asv5_train_protocol:
        train_datasets.append(
            ASVspoof5Dataset(args.asv5_train_root, args.asv5_train_protocol, subset="bonafide", **kw)
        )
    if args.mlaad_train_root and args.mlaad_train_protocol:
        train_datasets.append(
            MLAADDataset(args.mlaad_train_root, args.mlaad_train_protocol, subset="bonafide", **kw)
        )

    train_ds = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    total_train = sum(len(d) for d in train_datasets)
    print(f"Total training samples: {total_train} from {len(train_datasets)} dataset(s)")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    total_steps = len(train_loader) * cfg.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.lr,
        total_steps=total_steps,
        pct_start=0.1,
    )

    log_path = os.path.join(cfg.save_dir, "log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "lr"])

    best_val = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, cfg)
        val_loss = validate(model, dev_loader, device, cfg)
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch:3d} | train={train_loss:.5f} | val={val_loss:.5f} | lr={current_lr:.2e}")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss, current_lr])

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "config": cfg,
        }
        if val_loss < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save(ckpt, os.path.join(cfg.save_dir, "best.pt"))
        else:
            epochs_no_improve += 1
            if cfg.patience > 0 and epochs_no_improve >= cfg.patience:
                print(f"Early stopping: no improvement for {cfg.patience} epochs.")
                break

        if epoch % 5 == 0:
            torch.save(ckpt, os.path.join(cfg.save_dir, f"epoch_{epoch}.pt"))

    print(f"Training complete. Best val loss: {best_val:.5f}")


if __name__ == "__main__":
    main()
