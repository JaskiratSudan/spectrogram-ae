from .cnn_ae import CNNAutoencoder
from .convnext_ae import ConvNeXtAutoencoder
from .vit_ae import ViTAutoencoder
from .vit_small_ae import ViTSmallAutoencoder


def build_model(config):
    if config.encoder == "cnn":
        return CNNAutoencoder(latent_dim=config.latent_dim)
    if config.encoder == "convnext":
        return ConvNeXtAutoencoder(latent_dim=config.latent_dim)
    if config.encoder == "vit_tiny":
        return ViTAutoencoder(latent_dim=config.latent_dim)
    if config.encoder == "vit_small":
        return ViTSmallAutoencoder(latent_dim=config.latent_dim)
    raise ValueError(
        f"Unknown encoder: {config.encoder!r}. "
        "Choose 'cnn', 'convnext', 'vit_tiny', or 'vit_small'."
    )
