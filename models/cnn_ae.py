import torch
import torch.nn as nn


def _conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.GELU(),
        nn.MaxPool2d(2, 2),
    )


def _deconv_block(in_ch, out_ch, activation=True):
    layers = [
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, bias=False),
        nn.BatchNorm2d(out_ch),
    ]
    if activation:
        layers.append(nn.GELU())
    return nn.Sequential(*layers)


class CNNAutoencoder(nn.Module):
    """Small CNN autoencoder for mel spectrogram reconstruction.

    Input/output: [B, 1, 80, 500]
    """

    def __init__(self, latent_dim: int = 256):
        super().__init__()

        self.encoder_conv = nn.Sequential(
            _conv_block(1, 32),
            _conv_block(32, 64),
            _conv_block(64, 128),
            _conv_block(128, 256),
        )

        # Compute flattened size with a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 80, 500)
            out = self.encoder_conv(dummy)
            self.spatial_shape = out.shape[1:]   # (256, H', W')
            flat_size = out.numel()

        self.encoder_fc = nn.Linear(flat_size, latent_dim)

        self.decoder_fc = nn.Linear(latent_dim, flat_size)

        self.decoder_conv = nn.Sequential(
            _deconv_block(256, 128),
            _deconv_block(128, 64),
            _deconv_block(64, 32),
            # Last block: no GELU, output channel = 1
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2, bias=True),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 1, F, T] -> z: [B, latent_dim]"""
        h = self.encoder_conv(x)
        z = self.encoder_fc(h.flatten(1))
        return z

    def decode(self, z: torch.Tensor, spatial_shape=None) -> torch.Tensor:
        """z: [B, latent_dim] -> [B, 1, F, T]"""
        if spatial_shape is None:
            spatial_shape = self.spatial_shape
        c, h, w = spatial_shape
        h_flat = self.decoder_fc(z)
        h_3d = h_flat.view(-1, c, h, w)
        recon = self.decoder_conv(h_3d)
        return recon

    def forward(self, x: torch.Tensor):
        """x: [B, 1, F, T] -> (recon [B, 1, F, T], z [B, latent_dim])"""
        z = self.encode(x)
        recon = self.decode(z)
        # Match decoder output to input spatial size exactly (crop or pad)
        th, tw = x.shape[2], x.shape[3]
        rh, rw = recon.shape[2], recon.shape[3]
        if rh > th or rw > tw:
            recon = recon[:, :, :th, :tw]
        if recon.shape[2] < th or recon.shape[3] < tw:
            pad_h = th - recon.shape[2]
            pad_w = tw - recon.shape[3]
            recon = torch.nn.functional.pad(recon, (0, pad_w, 0, pad_h))
        return recon, z


# --- smoke test ---
if __name__ == "__main__":
    model = CNNAutoencoder(latent_dim=256)
    x = torch.randn(2, 1, 80, 500)
    recon, z = model(x)
    assert recon.shape == (2, 1, 80, 500), f"Expected (2,1,80,500), got {recon.shape}"
    assert z.shape == (2, 256), f"Expected (2,256), got {z.shape}"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"CNNAutoencoder smoke test passed: recon={recon.shape}, z={z.shape}, "
          f"params={n_params/1e6:.2f}M")
