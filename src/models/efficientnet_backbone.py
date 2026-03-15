"""
efficientnet_backbone.py
------------------------
Loads a pretrained EfficientNet-B0 backbone via timm.
Exposes freeze / unfreeze helpers used by the training loop.

Feature map size for EfficientNet-B0:
  Input  224×224  →  feature vector 1280-d  (after global avg pool)
"""

import timm
import torch
import torch.nn as nn


# ── Backbone ───────────────────────────────────────────────────────────────────

class EfficientNetBackbone(nn.Module):
    """
    EfficientNet-B0 with the classification head removed.

    forward() returns a (B, 1280) feature vector ready for the
    custom classifier head.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # num_classes=0 tells timm to drop the default head
        # global_pool=""  keeps the spatial feature map so we can
        # apply our own pooling in the classifier
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,       # remove default FC head
            global_pool="avg",   # keep timm's avg pool → (B, 1280)
        )

        self.feature_dim = self.backbone.num_features  # 1280

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, 1280) feature vectors."""
        return self.backbone(x)

    # ── Freeze / unfreeze helpers ──────────────────────────────────────────────

    def freeze_all(self):
        """Freeze every parameter (Phase-1 training: head only)."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_all(self):
        """Unfreeze all parameters (Phase-2 fine-tuning)."""
        for p in self.backbone.parameters():
            p.requires_grad = True

    def freeze_except_last_n_blocks(self, n: int = 3):
        """
        Partial unfreeze: keep early stem frozen, unfreeze the last N
        MBConv blocks + head. Good middle-ground if full fine-tuning
        overfits on small datasets.

        EfficientNet-B0 block names: blocks.0 … blocks.6  (7 blocks)
        """
        self.freeze_all()
        blocks = list(self.backbone.blocks.children())
        for block in blocks[-n:]:
            for p in block.parameters():
                p.requires_grad = True
        # Always unfreeze the BN + conv_head
        for p in self.backbone.conv_head.parameters():
            p.requires_grad = True
        for p in self.backbone.bn2.parameters():
            p.requires_grad = True

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = EfficientNetBackbone(pretrained=True).to(device)

    # Phase-1: frozen backbone
    model.freeze_all()
    print(f"Frozen   — trainable params: {model.trainable_params():,} / {model.total_params():,}")

    # Phase-2: full fine-tune
    model.unfreeze_all()
    print(f"Unfrozen — trainable params: {model.trainable_params():,} / {model.total_params():,}")

    # Forward pass
    dummy = torch.randn(4, 3, 224, 224).to(device)
    feats = model(dummy)
    print(f"Output shape: {feats.shape}")   # (4, 1280)
    assert feats.shape == (4, 1280), "Unexpected feature shape!"
    print("✅ Backbone OK")