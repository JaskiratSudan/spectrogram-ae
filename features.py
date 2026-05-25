import torch
import torchaudio.transforms as T

_MEL_CACHE: dict = {}
_AMP_TO_DB_CACHE: dict = {}

TARGET_FRAMES = 500


def compute_mel(waveform: torch.Tensor, config) -> torch.Tensor:
    """Compute log-mel spectrogram, fixed to [n_mels, 500].

    Args:
        waveform: [T] float tensor on any device
        config:   Config dataclass

    Returns:
        [n_mels, 500] dB-scale mel spectrogram
    """
    device = waveform.device
    key = (config.sample_rate, config.n_mels, config.n_fft,
           config.hop_length, config.win_length, config.f_min,
           config.f_max, str(device))

    if key not in _MEL_CACHE:
        _MEL_CACHE[key] = T.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            win_length=config.win_length,
            n_mels=config.n_mels,
            f_min=config.f_min,
            f_max=config.f_max,
        ).to(device)
        _AMP_TO_DB_CACHE[key] = T.AmplitudeToDB(stype="power", top_db=80).to(device)

    mel_transform = _MEL_CACHE[key]
    amp_to_db = _AMP_TO_DB_CACHE[key]

    mel = mel_transform(waveform)           # [n_mels, T']
    mel_db = amp_to_db(mel)                 # [n_mels, T']

    # Pad or crop to TARGET_FRAMES
    n_frames = mel_db.shape[-1]
    if n_frames < TARGET_FRAMES:
        pad_amt = TARGET_FRAMES - n_frames
        min_val = mel_db.min()
        mel_db = torch.nn.functional.pad(mel_db, (0, pad_amt), value=min_val.item())
    else:
        mel_db = mel_db[..., :TARGET_FRAMES]

    return mel_db  # [n_mels, 500]


def compute_contour_mask(mel_db: torch.Tensor, threshold_db: float = 40.0) -> torch.Tensor:
    """Compute binary edge mask via Sobel edge detection.

    Fully differentiable — no numpy, no .cpu() calls.

    Args:
        mel_db:       [F, T] dB-scale spectrogram
        threshold_db: edges above this percentile (scaled by /100) are kept

    Returns:
        [F, T] binary float mask (1 = strong edge, 0 = background)
    """
    device = mel_db.device

    # Sobel kernels
    kx = torch.tensor(
        [[-1., 0., 1.],
         [-2., 0., 2.],
         [-1., 0., 1.]],
        device=device
    ).view(1, 1, 3, 3)

    ky = torch.tensor(
        [[-1., -2., -1.],
         [ 0.,  0.,  0.],
         [ 1.,  2.,  1.]],
        device=device
    ).view(1, 1, 3, 3)

    x = mel_db.unsqueeze(0).unsqueeze(0)   # [1, 1, F, T]

    gx = torch.nn.functional.conv2d(x, kx, padding=1)  # [1, 1, F, T]
    gy = torch.nn.functional.conv2d(x, ky, padding=1)  # [1, 1, F, T]

    magnitude = torch.sqrt(gx ** 2 + gy ** 2).squeeze(0).squeeze(0)  # [F, T]

    # Normalize to [0, 1]
    mag_min = magnitude.min()
    mag_max = magnitude.max()
    normalized = (magnitude - mag_min) / (mag_max - mag_min + 1e-8)

    threshold = threshold_db / 100.0
    mask = (normalized > threshold).float()

    return mask  # [F, T]


def compute_energy_mask(mel_db: torch.Tensor, threshold_db: float = 40.0) -> torch.Tensor:
    """Binary mask keeping cells within threshold_db dB of the spectrogram's global max.

    Args:
        mel_db:       [F, T] dB-scale spectrogram
        threshold_db: keep pixels where value >= (global_max - threshold_db)

    Returns:
        [F, T] binary float mask
    """
    gate = mel_db.max() - threshold_db
    return (mel_db >= gate).float()


def compute_mask(mel_db: torch.Tensor, cfg) -> torch.Tensor:
    """Dispatch to the configured masking strategy."""
    if cfg.mask_type == "energy":
        return compute_energy_mask(mel_db, cfg.energy_threshold_db)
    return compute_contour_mask(mel_db, cfg.contour_threshold_db)


# --- smoke test ---
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from config import Config

    cfg = Config()
    device = torch.device("cpu")

    waveform = torch.randn(cfg.sample_rate * cfg.max_duration_seconds)
    mel = compute_mel(waveform, cfg)
    assert mel.shape == (80, 500), f"Expected (80, 500), got {mel.shape}"
    print(f"compute_mel smoke test passed: {mel.shape}")

    mask = compute_contour_mask(mel, cfg.contour_threshold_db)
    assert mask.shape == (80, 500), f"Expected (80, 500), got {mask.shape}"
    assert mask.min() >= 0.0 and mask.max() <= 1.0
    print(f"compute_contour_mask smoke test passed: {mask.shape}, "
          f"edge fraction={mask.mean().item():.3f}")

    emask = compute_energy_mask(mel, cfg.energy_threshold_db)
    assert emask.shape == (80, 500)
    print(f"compute_energy_mask smoke test passed: {emask.shape}, "
          f"energy fraction={emask.mean().item():.3f}")

    cfg.mask_type = "energy"
    m = compute_mask(mel, cfg)
    assert m.shape == (80, 500)
    cfg.mask_type = "contour"
    m = compute_mask(mel, cfg)
    assert m.shape == (80, 500)
    print("compute_mask dispatch smoke test passed.")
