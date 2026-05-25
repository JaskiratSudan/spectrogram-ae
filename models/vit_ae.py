import math
import torch
import torch.nn as nn


class ViTAutoencoder(nn.Module):
    """ViT-Tiny encoder + shallow transformer decoder for mel spectrogram reconstruction.

    Input/output: [B, 1, 80, 500]

    The encoder uses timm's vit_tiny_patch16_224, reconfigured for:
      - in_chans=1, img_size=img_size
      - classification head replaced with Identity
    """

    def __init__(
        self,
        latent_dim: int = 256,
        img_size: tuple[int, int] = (80, 500),
        patch_size: int = 16,
    ):
        super().__init__()
        import timm

        self.img_size = img_size
        self.patch_size = patch_size
        self.latent_dim = latent_dim

        # Pad img_size to nearest multiple of patch_size
        H, W = img_size
        self.pad_H = math.ceil(H / patch_size) * patch_size
        self.pad_W = math.ceil(W / patch_size) * patch_size
        self.n_patches_h = self.pad_H // patch_size
        self.n_patches_w = self.pad_W // patch_size
        self.n_patches = self.n_patches_h * self.n_patches_w  # N

        # Encoder backbone
        self.backbone = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=False,
            in_chans=1,
            img_size=(self.pad_H, self.pad_W),
            num_classes=0,   # removes classification head
        )
        self.d_model = self.backbone.embed_dim   # 192 for ViT-Tiny

        # Projection from backbone dim to latent_dim
        self.enc_proj = nn.Linear(self.d_model, latent_dim)

        # Decoder
        self.dec_input_proj = nn.Linear(latent_dim, self.d_model)
        self.dec_pos_embed = nn.Parameter(
            torch.zeros(1, self.n_patches, self.d_model)
        )
        nn.init.trunc_normal_(self.dec_pos_embed, std=0.02)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=3,
            dim_feedforward=self.d_model * 4,
            batch_first=True,
            norm_first=True,
        )
        self.decoder_transformer = nn.TransformerDecoder(decoder_layer, num_layers=4)

        # Reconstruct each patch: d_model -> patch_size * patch_size * 1
        self.patch_head = nn.Linear(self.d_model, patch_size * patch_size)

    def _pad_input(self, x: torch.Tensor) -> torch.Tensor:
        """Pad [B, 1, H, W] to [B, 1, pad_H, pad_W]."""
        _, _, H, W = x.shape
        pad_bottom = self.pad_H - H
        pad_right = self.pad_W - W
        if pad_bottom > 0 or pad_right > 0:
            x = torch.nn.functional.pad(x, (0, pad_right, 0, pad_bottom))
        return x

    def _unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        """Convert patch tokens back to image.

        patches: [B, N, patch_size*patch_size]
        returns: [B, 1, pad_H, pad_W]
        """
        B = patches.shape[0]
        p = self.patch_size
        nh, nw = self.n_patches_h, self.n_patches_w
        # [B, nh, nw, p, p]
        patches = patches.view(B, nh, nw, p, p)
        # [B, 1, nh*p, nw*p]
        img = patches.permute(0, 3, 1, 4, 2).contiguous().view(B, p * nh, p * nw)
        return img.unsqueeze(1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 1, F, T] -> z: [B, latent_dim]"""
        x_pad = self._pad_input(x)
        tokens = self.backbone.forward_features(x_pad)   # [B, N+1, D] with CLS
        # Drop CLS token if present, or use all patch tokens
        if tokens.shape[1] == self.n_patches + 1:
            patch_tokens = tokens[:, 1:, :]
        else:
            patch_tokens = tokens
        pooled = patch_tokens.mean(dim=1)                # [B, D]
        z = self.enc_proj(pooled)                        # [B, latent_dim]
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B, latent_dim] -> [B, 1, pad_H, pad_W]"""
        B = z.shape[0]
        h = self.dec_input_proj(z).unsqueeze(1)          # [B, 1, D]
        # Expand to N patch tokens + add positional embeddings
        query = h.expand(-1, self.n_patches, -1) + self.dec_pos_embed  # [B, N, D]
        # Use query as both tgt and memory for a lightweight decoder
        memory = h.expand(-1, self.n_patches, -1)
        decoded = self.decoder_transformer(query, memory)  # [B, N, D]
        patches = self.patch_head(decoded)                 # [B, N, p*p]
        recon = self._unpatchify(patches)                  # [B, 1, pad_H, pad_W]
        return recon

    def forward(self, x: torch.Tensor):
        """x: [B, 1, F, T] -> (recon [B, 1, F, T], z [B, latent_dim])"""
        z = self.encode(x)
        recon_padded = self.decode(z)
        # Crop back to original spatial size
        recon = recon_padded[:, :, : self.img_size[0], : self.img_size[1]]
        return recon, z


# --- smoke test ---
if __name__ == "__main__":
    model = ViTAutoencoder(latent_dim=256, img_size=(80, 500), patch_size=16)
    x = torch.randn(2, 1, 80, 500)
    recon, z = model(x)
    assert recon.shape == (2, 1, 80, 500), f"Expected (2,1,80,500), got {recon.shape}"
    assert z.shape == (2, 256), f"Expected (2,256), got {z.shape}"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ViTAutoencoder smoke test passed: recon={recon.shape}, z={z.shape}, "
          f"params={n_params/1e6:.2f}M")
