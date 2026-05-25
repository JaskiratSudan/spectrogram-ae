import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_eer(scores: list[float], labels: list[int]) -> tuple[float, float]:
    """Returns (eer_percent, threshold). Higher score = more likely spoof (label 0)."""
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=0)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[idx] + fnr[idx]) / 2.0 * 100.0
    return float(eer), float(thresholds[idx])


def save_error_map(error_map: np.ndarray, path: str) -> None:
    np.save(path, error_map)


def plot_error_map(
    mel_db: np.ndarray,
    error_map: np.ndarray,
    save_path: str,
    true_label: str = None,
    pred_label: str = None,
) -> None:
    """Two-panel figure: mel spectrogram (left) and error map overlay (right).

    Titles show ground-truth label on mel panel and predicted label on error panel.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    mel_title = "Mel Spectrogram"
    if true_label is not None:
        mel_title += f"\nGround truth: {true_label}"
    im0 = axes[0].imshow(mel_db, aspect="auto", origin="lower", cmap="viridis")
    axes[0].set_title(mel_title)
    axes[0].set_xlabel("Time frames")
    axes[0].set_ylabel("Mel bins")
    plt.colorbar(im0, ax=axes[0])

    err_title = "Reconstruction Error"
    if pred_label is not None:
        err_title += f"\nPredicted: {pred_label}"
    axes[1].imshow(mel_db, aspect="auto", origin="lower", cmap="viridis")
    im1 = axes[1].imshow(error_map, aspect="auto", origin="lower", cmap="RdYlGn_r", alpha=0.7)
    axes[1].set_title(err_title)
    axes[1].set_xlabel("Time frames")
    axes[1].set_ylabel("Mel bins")
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_paired_error_maps(
    pairs: list[tuple],
    save_path: str,
    threshold: float,
) -> None:
    """4×4 grid comparing real and fake paired utterances.

    pairs: list of (real_mel, real_err, real_score, fake_mel, fake_err, fake_score, pair_label)
           up to 4 pairs; pair_label is a short string like "Barack Obama #100"

    Layout (4 rows × 4 cols):
        Row 0: real mel spectrograms      (4 utterances)
        Row 1: real reconstruction errors (4 utterances)
        Row 2: fake mel spectrograms      (4 utterances)
        Row 3: fake reconstruction errors (4 utterances)
    """
    n = min(len(pairs), 4)
    fig, axes = plt.subplots(4, 4, figsize=(22, 14))

    row_labels = ["Real — Mel", "Real — Error", "Fake — Mel", "Fake — Error"]
    for row in range(4):
        axes[row][0].set_ylabel(row_labels[row], fontsize=10, labelpad=8)

    for col in range(n):
        real_mel, real_err, real_score, fake_mel, fake_err, fake_score, pair_label = pairs[col]

        real_pred = "bonafide" if real_score < threshold else "spoof"
        fake_pred = "bonafide" if fake_score < threshold else "spoof"

        # Row 0: real mel
        axes[0][col].imshow(real_mel, aspect="auto", origin="lower", cmap="viridis")
        axes[0][col].set_title(f"{pair_label}\nGT: bonafide | pred: {real_pred}", fontsize=8)
        axes[0][col].set_xticks([])
        axes[0][col].set_yticks([])

        # Row 1: real error overlay
        axes[1][col].imshow(real_mel, aspect="auto", origin="lower", cmap="viridis")
        axes[1][col].imshow(real_err, aspect="auto", origin="lower", cmap="RdYlGn_r", alpha=0.7)
        axes[1][col].set_title(f"score={real_score:.2f}", fontsize=8)
        axes[1][col].set_xticks([])
        axes[1][col].set_yticks([])

        # Row 2: fake mel
        axes[2][col].imshow(fake_mel, aspect="auto", origin="lower", cmap="viridis")
        axes[2][col].set_title(f"GT: spoof | pred: {fake_pred}", fontsize=8)
        axes[2][col].set_xticks([])
        axes[2][col].set_yticks([])

        # Row 3: fake error overlay
        axes[3][col].imshow(fake_mel, aspect="auto", origin="lower", cmap="viridis")
        axes[3][col].imshow(fake_err, aspect="auto", origin="lower", cmap="RdYlGn_r", alpha=0.7)
        axes[3][col].set_title(f"score={fake_score:.2f}", fontsize=8)
        axes[3][col].set_xticks([])
        axes[3][col].set_yticks([])

    # Hide unused columns
    for col in range(n, 4):
        for row in range(4):
            axes[row][col].set_visible(False)

    plt.suptitle("Real vs Fake Paired Comparison", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_score_distribution(
    scores: list,
    labels: list,
    save_path: str,
    dataset_name: str,
    threshold: float = None,
) -> None:
    """Histogram of reconstruction error scores split by bonafide vs spoof."""
    bona_scores = [s for s, l in zip(scores, labels) if l == 1]
    spoof_scores = [s for s, l in zip(scores, labels) if l == 0]

    fig, ax = plt.subplots(figsize=(8, 4))

    bins = np.linspace(
        min(scores) if scores else 0,
        max(scores) if scores else 1,
        60,
    )
    if bona_scores:
        ax.hist(bona_scores, bins=bins, alpha=0.6, color="steelblue", label=f"Bonafide (n={len(bona_scores)})")
    if spoof_scores:
        ax.hist(spoof_scores, bins=bins, alpha=0.6, color="tomato", label=f"Spoof (n={len(spoof_scores)})")
    if threshold is not None:
        ax.axvline(threshold, color="black", linestyle="--", linewidth=1.2, label=f"EER threshold={threshold:.3f}")

    ax.set_xlabel("Reconstruction Error Score")
    ax.set_ylabel("Count")
    ax.set_title(f"{dataset_name} — Score Distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def set_seed(seed: int = 42) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --- smoke test ---
if __name__ == "__main__":
    import tempfile, os

    scores = [0.1, 0.2, 0.8, 0.9, 0.15, 0.85]
    labels = [1, 1, 0, 0, 1, 0]
    eer, thresh = compute_eer(scores, labels)
    assert 0.0 <= eer <= 100.0
    print(f"compute_eer smoke test passed: EER={eer:.2f}%, threshold={thresh:.4f}")

    mel = np.random.randn(80, 500).astype(np.float32)
    err = np.abs(np.random.randn(80, 500).astype(np.float32))
    with tempfile.TemporaryDirectory() as tmpdir:
        # Individual plot with labels
        png = os.path.join(tmpdir, "test.png")
        plot_error_map(mel, err, png, true_label="bonafide", pred_label="spoof")
        assert os.path.exists(png)
        print("plot_error_map smoke test passed.")

        # Paired plot
        pairs = [(mel, err, 5.0, mel * 0.5, err * 2, 12.0, "Speaker #1")] * 4
        png2 = os.path.join(tmpdir, "paired.png")
        plot_paired_error_maps(pairs, png2, threshold=8.0)
        assert os.path.exists(png2)
        print("plot_paired_error_maps smoke test passed.")

    set_seed(42)
    print("set_seed smoke test passed.")
