"""
classifier.py
-------------
Full model: EfficientNet-B0 backbone + custom classification head.

Architecture:
  Input (B,3,224,224)
    → EfficientNetBackbone  → (B, 1280)
    [optional CBAM attention]
    → Dropout(0.4)
    → Linear(1280 → 512) + ReLU + BatchNorm
    → Dropout(0.3)
    → Linear(512 → 43)
    → logits  (B, 43)

Usage:
  model = get_model(config)
  logits = model(images)          # training
  probs  = model.predict(images)  # inference (returns softmax)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.efficientnet_backbone import EfficientNetBackbone


# ── Optional: CBAM Attention ───────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention."""

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        mid = max(in_channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C) — already after global avg pool
        return x * torch.sigmoid(self.fc(x))


# ── Classifier Head ────────────────────────────────────────────────────────────

class ClassifierHead(nn.Module):
    """
    Custom head that sits on top of the 1280-d feature vector.

    Parameters
    ----------
    in_features  : Feature dim from backbone (1280 for EfficientNet-B0).
    num_classes  : Number of output classes (43 for GTSRB).
    hidden_dim   : Intermediate FC layer width.
    dropout1     : Dropout after backbone features.
    dropout2     : Dropout after hidden layer.
    use_attention: If True, applies channel attention before FC layers.
    """

    def __init__(
        self,
        in_features: int = 1280,
        num_classes: int = 43,
        hidden_dim: int  = 512,
        dropout1: float  = 0.4,
        dropout2: float  = 0.3,
        use_attention: bool = True,
    ):
        super().__init__()
        self.use_attention = use_attention

        if use_attention:
            self.attention = ChannelAttention(in_features)

        self.head = nn.Sequential(
            nn.Dropout(p=dropout1),
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout2),
            nn.Linear(hidden_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_attention:
            x = self.attention(x)
        return self.head(x)


# ── Full Model ─────────────────────────────────────────────────────────────────

class TrafficSignClassifier(nn.Module):
    """
    End-to-end model combining backbone + head.

    Training phases
    ---------------
    Phase 1 (frozen backbone):
        Only the head is trained. Call model.set_phase(1).
    Phase 2 (full fine-tune):
        Everything is trained with a lower LR. Call model.set_phase(2).
    """

    def __init__(
        self,
        num_classes: int  = 43,
        hidden_dim: int   = 512,
        dropout1: float   = 0.4,
        dropout2: float   = 0.3,
        pretrained: bool  = True,
        use_attention: bool = True,
    ):
        super().__init__()

        self.backbone = EfficientNetBackbone(pretrained=pretrained)
        self.head = ClassifierHead(
            in_features=self.backbone.feature_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout1=dropout1,
            dropout2=dropout2,
            use_attention=use_attention,
        )

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits (B, num_classes)."""
        features = self.backbone(x)
        return self.head(features)

    @torch.inference_mode()
    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Inference-mode forward pass.

        Returns
        -------
        probs  : (B, num_classes) softmax probabilities
        preds  : (B,) predicted class indices
        """
        self.eval()
        logits = self(x)
        probs  = F.softmax(logits, dim=1)
        preds  = probs.argmax(dim=1)
        return probs, preds

    # ── Phase helpers ──────────────────────────────────────────────────────────

    def set_phase(self, phase: int):
        """
        phase=1 → freeze backbone, train head only.
        phase=2 → unfreeze everything for fine-tuning.
        """
        if phase == 1:
            self.backbone.freeze_all()
            print("📌 Phase 1: backbone frozen — training head only.")
        elif phase == 2:
            self.backbone.unfreeze_all()
            print("🔓 Phase 2: backbone unfrozen — full fine-tuning.")
        else:
            raise ValueError(f"phase must be 1 or 2, got {phase}")

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def param_summary(self):
        trainable = self.trainable_params()
        total     = self.total_params()
        print(f"  Total params     : {total:,}")
        print(f"  Trainable params : {trainable:,}")
        print(f"  Frozen params    : {total - trainable:,}")


# ── Factory function ───────────────────────────────────────────────────────────

def get_model(config: dict) -> TrafficSignClassifier:
    """
    Build the model from a loaded train_config.yaml dict.

    Usage:
        from src.augmentation import load_config
        from src.models.classifier import get_model
        config = load_config()
        model  = get_model(config)
    """
    m = config["model"]
    return TrafficSignClassifier(
        num_classes   = config["data"]["num_classes"],
        hidden_dim    = m["hidden_dim"],
        dropout1      = m["dropout1"],
        dropout2      = m["dropout2"],
        pretrained    = m["pretrained"],
        use_attention = True,
    )


def get_device(config: dict) -> torch.device:
    """Resolve device from config ('auto', 'cpu', 'cuda', 'mps')."""
    pref = config.get("inference", {}).get("device", "auto")
    if pref == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(pref)


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.augmentation import load_config

    config = load_config()
    device = get_device(config)
    print(f"\nDevice: {device}")

    model = get_model(config).to(device)

    # Phase 1 — head only
    model.set_phase(1)
    model.param_summary()

    # Phase 2 — full fine-tune
    model.set_phase(2)
    model.param_summary()

    # Forward pass
    dummy  = torch.randn(4, 3, 224, 224).to(device)
    logits = model(dummy)
    print(f"\nLogits shape : {logits.shape}")   # (4, 43)

    # Predict
    probs, preds = model.predict(dummy)
    print(f"Probs shape  : {probs.shape}")      # (4, 43)
    print(f"Predictions  : {preds.tolist()}")
    print(f"Confidence   : {probs.max(dim=1).values.tolist()}")

    assert logits.shape == (4, 43)
    print("\n✅ Classifier OK")