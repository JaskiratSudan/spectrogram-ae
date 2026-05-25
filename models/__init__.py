from .cnn_ae import CNNAutoencoder
from .vit_ae import ViTAutoencoder


def build_model(config):
    if config.encoder == "cnn":
        return CNNAutoencoder(latent_dim=config.latent_dim)
    elif config.encoder == "vit_tiny":
        return ViTAutoencoder(latent_dim=config.latent_dim)
    else:
        raise ValueError(f"Unknown encoder: {config.encoder!r}. Choose 'cnn' or 'vit_tiny'.")
