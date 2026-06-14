import argparse
import csv
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config
from dataset import (
    ASVspoof2019Dataset,
    ASVspoof5Dataset,
    ATADDDataset,
    FakeXposeDataset,
    FamousFiguresDataset,
    InTheWildDataset,
    MLAADDataset,
    collate_fn,
)
from features import compute_mask, compute_mel
from models import build_model
from utils import compute_eer, plot_error_map, plot_paired_error_maps, plot_score_distribution

# Datasets known to have paired real/fake utterances
PAIRED_DATASETS = {"fakexpose", "famousfigures"}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate spectrogram autoencoder")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--dataset", required=True,
        choices=["asv19", "itw", "fakexpose", "asv5", "mlaad", "famousfigures", "atadd", "all"],
    )
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--root", default=None)
    parser.add_argument("--subset", default="all")
    parser.add_argument("--results_dir", default=None,
                        help="Directory to write scores/plots into. "
                             "Defaults to results/<encoder> if not set.")
    # Feature-space scoring
    parser.add_argument("--score_type", default="pixel", choices=["pixel", "feature"],
                        help="pixel: mean |mel-recon| (default); "
                             "feature: weighted L2 in encoder feature space (CNN/ConvNeXt only)")
    parser.add_argument("--feature_weights", default="1.0,0.8,0.4,0.2",
                        help="Comma-separated per-block weights for feature scoring "
                             "(index 0 = shallowest/finest). Default: 1.0,0.8,0.4,0.2")
    parser.add_argument("--no_normalize_features", action="store_true",
                        help="Disable per-channel L2 normalization before feature distance")
    return parser.parse_args()


def load_checkpoint(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg: Config = ckpt["config"]
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def _normalize_feat(f: torch.Tensor) -> torch.Tensor:
    """Normalize each channel to unit L2 norm across spatial dims [B, C, H, W]."""
    norm = f.norm(dim=(2, 3), keepdim=True).clamp(min=1e-8)
    return f / norm


def _compute_feature_score(model, mel_4d: torch.Tensor, recon_4d: torch.Tensor,
                            weights: tuple, normalize: bool) -> float:
    """Weighted L2 distance in encoder feature space.

    mel_4d / recon_4d: [1, 1, F, T]
    weights: one scalar per encoder block (index 0 = finest/shallowest)
    Returns scalar score.
    """
    feats_orig  = model.intermediate_features(mel_4d)
    feats_recon = model.intermediate_features(recon_4d)
    score = 0.0
    for w, f_o, f_r in zip(weights, feats_orig, feats_recon):
        if normalize:
            f_o = _normalize_feat(f_o)
            f_r = _normalize_feat(f_r)
        dist = (f_o - f_r).pow(2).sum(dim=1).mean().item()  # sum over C, mean over H,W
        score += w * dist
    return score


def _score_waveform(model, cfg, device, waveform: torch.Tensor, score_cfg: dict = None):
    """Compute (mel_np, recon_np, error_np, score) for a single waveform tensor [T]."""
    mel = compute_mel(waveform.to(device), cfg)
    mask = compute_mask(mel, cfg)
    x_in = (mel * mask).unsqueeze(0).unsqueeze(0)   # [1, 1, F, T]
    recon_4d = model(x_in)[0]                        # [1, 1, F, T]
    recon = recon_4d.squeeze()                        # [F, T]
    error_map = (mel - recon).abs()

    use_feature = (
        score_cfg is not None
        and score_cfg.get("score_type") == "feature"
        and hasattr(model, "intermediate_features")
    )
    if use_feature:
        mel_4d = mel.unsqueeze(0).unsqueeze(0)
        score = _compute_feature_score(
            model, mel_4d, recon_4d,
            weights=score_cfg["feature_weights"],
            normalize=score_cfg["normalize"],
        )
    else:
        score = error_map.mean().item()

    return mel.cpu().numpy(), recon.cpu().numpy(), error_map.cpu().numpy(), score


def _build_name_to_idx(dataset, dataset_name: str):
    """Return two dicts: bonafide_name→idx and spoof_name→idx into dataset."""
    bona, spoof = {}, {}
    if dataset_name == "fakexpose":
        for i, (path, label) in enumerate(dataset.samples):
            name = Path(path).name
            (bona if label == 1 else spoof)[name] = i
    else:
        # All other datasets: samples = [(audio_name, label, path), ...]
        for i, s in enumerate(dataset.samples):
            name, label = s[0], s[1]
            (bona if label == 1 else spoof)[name] = i
    return bona, spoof


def _find_pairs(all_results: list, dataset_name: str) -> list[tuple[str, str]]:
    """Find (real_name, fake_name) pairs from scored samples.

    FakeXpose:      same filename exists as bonafide AND spoof
    FamousFigures:  fake name ends with __REAL_NAME; match against bonafide names
    """
    bona_names = {name for _, label, name in all_results if label == 1}
    spoof_names = {name for _, label, name in all_results if label == 0}
    pairs = []

    if dataset_name == "fakexpose":
        for name in spoof_names:
            if name in bona_names:
                pairs.append((name, name))

    elif dataset_name == "famousfigures":
        for fake_name in spoof_names:
            if "__" in fake_name:
                real_id = fake_name.split("__")[-1]
                if real_id in bona_names:
                    pairs.append((real_id, fake_name))

    return pairs


def _pair_label(real_name: str, fake_name: str, dataset_name: str) -> str:
    """Short human-readable label for a pair."""
    if dataset_name == "fakexpose":
        stem = Path(real_name).stem
        parts = stem.rsplit("_", 1)
        return f"{parts[0].replace('_', ' ')} #{parts[1]}" if len(parts) == 2 else stem
    elif dataset_name == "famousfigures":
        real_id = fake_name.split("__")[-1]
        speaker = real_id.rsplit("_", 1)[0].replace("_", " ")
        return speaker
    return real_name


@torch.no_grad()
def evaluate_single(model, cfg, device, dataset_name: str, dataset, results_dir: str,
                    score_cfg: dict = None) -> dict:
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f"{dataset_name}_scores.csv")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # Scalar records for ALL samples — used for EER, pair detection
    all_results: list[tuple[float, int, str]] = []
    # Reservoir of (mel_np, recon_np, err_np, audio_name, label, score)
    reservoir: list = []
    RESERVOIR_SIZE = 10

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["audio_name", "label", "score"])
        for i, (waveforms, label_batch, audio_names) in enumerate(loader):
            waveform = waveforms[0]
            label = label_batch[0].item()
            audio_name = audio_names[0]

            mel_np, recon_np, err_np, score = _score_waveform(
                model, cfg, device, waveform, score_cfg)
            writer.writerow([audio_name, label, f"{score:.6f}"])
            all_results.append((score, label, audio_name))

            # Reservoir sampling
            if len(reservoir) < RESERVOIR_SIZE:
                reservoir.append((mel_np, recon_np, err_np, audio_name, label, score))
            else:
                j = random.randint(0, i)
                if j < RESERVOIR_SIZE:
                    reservoir[j] = (mel_np, recon_np, err_np, audio_name, label, score)

    print(f"Scores saved to {csv_path}")

    # Metrics
    scores = [r[0] for r in all_results]
    labels = [r[1] for r in all_results]
    try:
        from sklearn.metrics import roc_auc_score
        auroc = roc_auc_score(labels, [-s for s in scores]) * 100.0
    except Exception:
        auroc = float("nan")
    eer, threshold = compute_eer(scores, labels)
    print(f"[{dataset_name}] EER={eer:.2f}%  AUROC={auroc:.2f}%  "
          f"threshold={threshold:.4f}  n={len(scores)}")
    print()

    # Score distribution plot
    dist_path = os.path.join(results_dir, f"{dataset_name}_score_dist.png")
    plot_score_distribution(scores, labels, dist_path, dataset_name, threshold)
    print(f"[{dataset_name}] Score distribution saved to {dist_path}")

    # Individual sample plots with labels
    name_to_score = {name: score for score, _, name in all_results}
    for mel_np, recon_np, err_np, audio_name, label, score in reservoir:
        true_label = "bonafide" if label == 1 else "spoof"
        pred_label = "bonafide" if score < threshold else "spoof"
        safe = audio_name.replace("/", "_").replace(".", "_")
        png_path = os.path.join(results_dir, f"{dataset_name}_{safe}_error.png")
        plot_error_map(mel_np, recon_np, err_np, png_path, true_label=true_label, pred_label=pred_label)

    # Paired plots for applicable datasets
    if dataset_name in PAIRED_DATASETS:
        _make_paired_plot(
            model, cfg, device, dataset, dataset_name,
            all_results, name_to_score, threshold, results_dir, score_cfg,
        )

    return {"dataset": dataset_name, "eer": eer, "auroc": auroc,
            "threshold": threshold, "n": len(scores)}


@torch.no_grad()
def _make_paired_plot(model, cfg, device, dataset, dataset_name,
                      all_results, name_to_score, threshold, results_dir, score_cfg=None):
    """Find 4 matched real/fake pairs, load their mels, save 4×4 comparison plot."""
    pairs = _find_pairs(all_results, dataset_name)
    if not pairs:
        print(f"[{dataset_name}] No paired utterances found for paired plot.")
        return

    random.shuffle(pairs)
    chosen = pairs[:4]

    # Build name→dataset_idx for direct O(1) sample access
    bona_idx, spoof_idx = _build_name_to_idx(dataset, dataset_name)

    plot_pairs = []
    for real_name, fake_name in chosen:
        r_idx = bona_idx.get(real_name)
        f_idx = spoof_idx.get(fake_name)
        if r_idx is None or f_idx is None:
            continue

        r_wave, _, _ = dataset[r_idx]
        f_wave, _, _ = dataset[f_idx]

        r_mel, r_recon, r_err, r_score = _score_waveform(model, cfg, device, r_wave, score_cfg)
        f_mel, f_recon, f_err, f_score = _score_waveform(model, cfg, device, f_wave, score_cfg)

        label = _pair_label(real_name, fake_name, dataset_name)
        plot_pairs.append((r_mel, r_recon, r_err, r_score, f_mel, f_recon, f_err, f_score, label))

    if plot_pairs:
        save_path = os.path.join(results_dir, f"{dataset_name}_paired_comparison.png")
        plot_paired_error_maps(plot_pairs, save_path, threshold)
        print(f"[{dataset_name}] Paired plot saved to {save_path}")


def build_dataset(dataset_name: str, args, cfg: Config):
    kw = dict(target_sample_rate=cfg.sample_rate, max_duration_seconds=cfg.max_duration_seconds)
    if dataset_name == "asv19":
        return ASVspoof2019Dataset(args.root, args.protocol, subset=args.subset, **kw)
    if dataset_name == "asv5":
        return ASVspoof5Dataset(args.root, args.protocol, subset=args.subset, **kw)
    if dataset_name == "itw":
        return InTheWildDataset(args.root, args.protocol, subset=args.subset, **kw)
    if dataset_name == "fakexpose":
        return FakeXposeDataset(args.root, subset=args.subset, **kw)
    if dataset_name == "mlaad":
        return MLAADDataset(args.root, args.protocol, subset=args.subset, **kw)
    if dataset_name == "famousfigures":
        return FamousFiguresDataset(args.root, args.protocol, subset=args.subset, **kw)
    if dataset_name == "atadd":
        return ATADDDataset(args.root, args.protocol, subset=args.subset, **kw)
    raise ValueError(f"Unknown dataset: {dataset_name}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_checkpoint(args.checkpoint, device)
    results_dir = args.results_dir if args.results_dir else os.path.join("results", cfg.encoder)

    score_cfg = {
        "score_type": args.score_type,
        "feature_weights": tuple(float(w) for w in args.feature_weights.split(",")),
        "normalize": not args.no_normalize_features,
    }
    if score_cfg["score_type"] == "feature" and not hasattr(model, "intermediate_features"):
        print(f"[WARN] --score_type=feature not supported for encoder '{cfg.encoder}'; "
              "falling back to pixel scoring.")
        score_cfg["score_type"] = "pixel"

    print(f"Scoring mode : {score_cfg['score_type']}")
    if score_cfg["score_type"] == "feature":
        print(f"  weights    : {score_cfg['feature_weights']}")
        print(f"  normalize  : {score_cfg['normalize']}")

    if args.dataset == "all":
        summary = []
        for name in ["asv19", "asv5", "itw", "fakexpose", "mlaad", "famousfigures"]:
            if args.root is None:
                print(f"Skipping {name}: --root not provided")
                continue
            try:
                ds = build_dataset(name, args, cfg)
                row = evaluate_single(model, cfg, device, name, ds, results_dir, score_cfg)
                summary.append(row)
            except Exception as e:
                print(f"[{name}] skipped: {e}")

        print("\n--- Summary ---")
        print(f"{'Dataset':<16} {'EER':>8} {'AUROC':>8} {'N':>8}")
        for row in summary:
            print(f"{row['dataset']:<16} {row['eer']:>7.2f}% {row['auroc']:>7.2f}% {row['n']:>8}")
    else:
        ds = build_dataset(args.dataset, args, cfg)
        evaluate_single(model, cfg, device, args.dataset, ds, results_dir, score_cfg)


if __name__ == "__main__":
    main()
