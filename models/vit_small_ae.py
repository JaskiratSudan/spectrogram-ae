import torch
import torch.nn as nn
import torch.nn.functional as F


class ViTSmallAutoencoder(nn.Module):
    """ViT-Small autoencoder with asymmetric 4×16 mel patches.

    Patch size 4 (freq) × 16 (time) gives 20×32 = 640 patches per spectrogram,
    providing much finer frequency resolution than a standard 16×16 grid.
    Input time axis is zero-padded to 512 before encoding and cropped after decoding.

    Encoder: 12-layer transformer  (embed_dim=384, heads=6, ff=1536)
    Decoder:  4-layer transformer  → patch-wise linear → unpatchify

    Input/output: [B, 1, 80, 500]
    """

    PATCH_F: int = 4
    PATCH_T: int = 16
    PAD_T: int = 512          # pad time to 512 so 512 / 16 = 32 is exact
    N_PF: int = 80 // 4       # 20 patches along frequency
    N_PT: int = 512 // 16     # 32 patches along time
    N_PATCHES: int = 20 * 32  # 640
    D_MODEL: int = 384

    def __init__(self, latent_dim: int = 256):
        super().__init__()

        # --- Encoder ---
        self.patch_embed = nn.Conv2d(
            1, self.D_MODEL,
            kernel_size=(self.PATCH_F, self.PATCH_T),
            stride=(self.PATCH_F, self.PATCH_T),
        )
        self.enc_pos_embed = nn.Parameter(torch.zeros(1, self.N_PATCHES, self.D_MODEL))
        nn.init.trunc_normal_(self.enc_pos_embed, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.D_MODEL, nhead=6, dim_feedforward=1536,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer_enc = nn.TransformerEncoder(
            enc_layer, num_layers=12, norm=nn.LayerNorm(self.D_MODEL),
        )
        self.enc_proj = nn.Linear(self.D_MODEL, latent_dim)

        # --- Decoder ---
        self.dec_proj = nn.Linear(latent_dim, self.D_MODEL)
        self.dec_pos_embed = nn.Parameter(torch.zeros(1, self.N_PATCHES, self.D_MODEL))
        nn.init.trunc_normal_(self.dec_pos_embed, std=0.02)

        dec_layer = nn.TransformerEncoderLayer(
            d_model=self.D_MODEL, nhead=6, dim_feedforward=1536,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer_dec = nn.TransformerEncoder(
            dec_layer, num_layers=4, norm=nn.LayerNorm(self.D_MODEL),
        )
        self.patch_pred = nn.Linear(self.D_MODEL, self.PATCH_F * self.PATCH_T)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 1, 80, T] -> z: [B, latent_dim]"""
        x = F.pad(x, (0, self.PAD_T - x.shape[-1]))  # [B, 1, 80, 512]
        tokens = self.patch_embed(x)                   # [B, D, 20, 32]
        tokens = tokens.flatten(2).transpose(1, 2)    # [B, 640, D]
        tokens = tokens + self.enc_pos_embed
        tokens = self.transformer_enc(tokens)          # [B, 640, D]
        return self.enc_proj(tokens.mean(dim=1))       # [B, latent_dim]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B, latent_dim] -> recon: [B, 1, 80, 512]"""
        B = z.shape[0]
        tokens = self.dec_proj(z).unsqueeze(1).expand(-1, self.N_PATCHES, -1)  # [B, 640, D]
        tokens = tokens + self.dec_pos_embed
        tokens = self.transformer_dec(tokens)         # [B, 640, D]
        patches = self.patch_pred(tokens)             # [B, 640, 64]

        # unpatchify: [B, 20, 32, 4, 16] → [B, 1, 80, 512]
        patches = patches.view(B, self.N_PF, self.N_PT, self.PATCH_F, self.PATCH_T)
        recon = patches.permute(0, 1, 3, 2, 4).contiguous()
        recon = recon.view(B, 1, self.N_PF * self.PATCH_F, self.N_PT * self.PATCH_T)
        return recon

    def forward(self, x: torch.Tensor):
        """x: [B, 1, 80, 500] -> (recon [B, 1, 80, 500], z [B, latent_dim])"""
        orig_T = x.shape[-1]
        z = self.encode(x)
        recon = self.decode(z)[:, :, :, :orig_T]   # crop padded time back to 500
        return recon, z


# --- smoke test ---
if __name__ == "__main__":
    model = ViTSmallAutoencoder(latent_dim=256)
    x = torch.randn(2, 1, 80, 500)
    recon, z = model(x)
    assert recon.shape == (2, 1, 80, 500), f"Expected (2,1,80,500), got {recon.shape}"
    assert z.shape == (2, 256), f"Expected (2,256), got {z.shape}"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ViTSmallAutoencoder smoke test passed: recon={recon.shape}, z={z.shape}, "
          f"params={n_params/1e6:.2f}M")
