from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    # Audio
    sample_rate: int = 16000
    max_duration_seconds: int = 5

    # Mel spectrogram
    n_mels: int = 80
    n_fft: int = 1024
    hop_length: int = 160
    win_length: int = 400
    f_min: float = 0.0
    f_max: float = 8000.0

    # Masking strategy: "contour" (Sobel edges) or "energy" (dB-below-max gate)
    mask_type: str = "contour"
    contour_threshold_db: float = 40.0   # keep edges where norm magnitude > threshold/100
    energy_threshold_db: float = 40.0    # keep cells within this many dB of the global max

    # Model
    encoder: str = "cnn"        # "cnn" or "vit_tiny"
    latent_dim: int = 256

    # Training
    batch_size: int = 32
    lr: float = 1e-3
    epochs: int = 50
    patience: int = 10
    rawboost_prob: float = 0.7
    num_workers: int = 4
    device: str = "cuda"
    save_dir: str = "checkpoints"

    # Evaluation
    eval_threshold: Optional[float] = None


if __name__ == "__main__":
    cfg = Config()
    print(cfg)
    assert cfg.sample_rate == 16000
    assert cfg.encoder == "cnn"
    print("Config smoke test passed.")
