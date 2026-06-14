import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNeXtBlock(nn.Module):
    """Depthwise 7×7 → LayerNorm → inverted-bottleneck MLP → residual."""

    def __init__(self, dim: int, expansion: int = 4):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Linear(dim, dim * expansion)
        self.act = nn.GELU()
        self.pw2 = nn.Linear(dim * expansion, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dw(x)
        x = x.permute(0, 2, 3, 1)   # [B, H, W, C] for LayerNorm over channels
        x = self.norm(x)
        x = self.pw2(self.act(self.pw1(x)))
        x = x.permute(0, 3, 1, 2)   # [B, C, H, W]
        return x + shortcut


def _cnx_stage(in_ch: int, out_ch: int) -> nn.Sequential:
    """1×1 channel projection → BN → ConvNeXt block → MaxPool."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=1),
        nn.BatchNorm2d(out_ch),
        ConvNeXtBlock(out_ch),
        nn.MaxPool2d(2, 2),
    )


def _deconv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.GELU(),
    )


class ConvNeXtAutoencoder(nn.Module):
    """ConvNeXt-style mel spectrogram autoencoder.

    Same channel progression and spatial layout as CNNAutoencoder (1→32→64→128→256,
    four MaxPool 2×2 stages), but each stage uses depthwise 7×7 convolutions with
    an inverted-bottleneck MLP and residual connections instead of Conv-BN-GELU.

    Input/output: [B, 1, 80, 500]
    """

    def __init__(self, latent_dim: int = 256):
        super().__init__()

        self.encoder_conv = nn.Sequential(
            _cnx_stage(1, 32),
            _cnx_stage(32, 64),
            _cnx_stage(64, 128),
            _cnx_stage(128, 256),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, 80, 500)
            out = self.encoder_conv(dummy)
            self.spatial_shape = out.shape[1:]
            flat_size = out.numel()

        self.encoder_fc = nn.Linear(flat_size, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, flat_size)

        self.decoder_conv = nn.Sequential(
            _deconv_block(256, 128),
            _deconv_block(128, 64),
            _deconv_block(64, 32),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2, bias=True),
        )

    def _encode_blocks(self, x: torch.Tensor):
        """Run encoder block-by-block, returning all 4 intermediate feature maps."""
        f1 = self.encoder_conv[0](x)
        f2 = self.encoder_conv[1](f1)
        f3 = self.encoder_conv[2](f2)
        f4 = self.encoder_conv[3](f3)
        return f1, f2, f3, f4

    def intermediate_features(self, x: torch.Tensor):
        """Return (f1, f2, f3, f4) feature maps for feature-space scoring.

        f1: [B, 32,  40, 250]  — finest, local texture
        f4: [B, 256,  5,  31]  — coarsest, global structure
        """
        return self._encode_blocks(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder_conv(x)
        return self.encoder_fc(h.flatten(1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        c, h, w = self.spatial_shape
        h_3d = self.decoder_fc(z).view(-1, c, h, w)
        return self.decoder_conv(h_3d)

    def forward(self, x: torch.Tensor):
        """x: [B, 1, F, T] -> (recon [B, 1, F, T], z [B, latent_dim])"""
        z = self.encode(x)
        recon = self.decode(z)
        th, tw = x.shape[2], x.shape[3]
        if recon.shape[2] > th or recon.shape[3] > tw:
            recon = recon[:, :, :th, :tw]
        if recon.shape[2] < th or recon.shape[3] < tw:
            recon = F.pad(recon, (0, tw - recon.shape[3], 0, th - recon.shape[2]))
        return recon, z


# --- smoke test ---
if __name__ == "__main__":
    model = ConvNeXtAutoencoder(latent_dim=256)
    x = torch.randn(2, 1, 80, 500)
    recon, z = model(x)
    assert recon.shape == (2, 1, 80, 500), f"Expected (2,1,80,500), got {recon.shape}"
    assert z.shape == (2, 256), f"Expected (2,256), got {z.shape}"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ConvNeXtAutoencoder smoke test passed: recon={recon.shape}, z={z.shape}, "
          f"params={n_params/1e6:.2f}M")
