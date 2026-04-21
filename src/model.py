import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class ResNetBackbone(nn.Module):
    """ResNet50 backbone returning a feature map (B, C, H, W).
    Removes avgpool and fc.
    """

    def __init__(self, pretrained=False):
        super().__init__()
        resnet = torchvision.models.resnet50(weights=None)
        # keep layers up to layer4 (exclude avgpool & fc)
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class ResNet50Backbone(nn.Sequential):
    """ResNet50 backbone returning a feature map (B, 2048, H, W).
    Matches the feature extractor part of torchvision ResNet50 up to (and including) layer4.

    Note: This is intentionally implemented as an nn.Sequential of ResNet children
    to make it easy to load SSL checkpoints that store keys like `backbone.0.weight`, etc.
    """

    def __init__(self, pretrained_imagenet=False):
        weights = torchvision.models.ResNet50_Weights.DEFAULT if pretrained_imagenet else None
        resnet = torchvision.models.resnet50(weights=weights)
        # keep layers up to layer4 (exclude avgpool & fc)
        super().__init__(*list(resnet.children())[:-2])
        if pretrained_imagenet:
            print("[INFO] Loaded ImageNet pretrained weights for ResNet50 backbone")


class TransformerHead(nn.Module):
    def __init__(
        self,
        in_channels=512,
        d_model=512,
        nhead=8,
        num_layers=4,
        dropout=0.1,
        input_size=128,
    ):
        super().__init__()
        self.proj = nn.Linear(in_channels, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # For input_size=128 with ResNet50 (stride 32), feature map is 4x4 = 16 tokens + 1 cls = 17
        # Pre-compute expected sequence length for proper initialization
        feat_size = input_size // 32  # ResNet50 downsamples by 32x
        num_tokens = feat_size * feat_size + 1  # +1 for CLS token
        self.register_buffer("_num_tokens", torch.tensor(num_tokens))
        self.pos_embed = nn.Parameter(torch.randn(1, num_tokens, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, feat):
        # feat: (B, C, H, W)
        B, C, H, W = feat.shape
        x = feat.flatten(2).transpose(1, 2)  # (B, N, C) where N=H*W
        x = self.proj(x)  # (B, N, d_model)

        N = x.shape[1]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B,1,d)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, d)

        # Handle case where input size differs from expected (rare, for flexibility)
        if self.pos_embed.shape[1] != N + 1:
            # Interpolate positional embeddings if needed
            pos_embed = torch.nn.functional.interpolate(
                self.pos_embed.transpose(1, 2),
                size=N + 1,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        else:
            pos_embed = self.pos_embed

        x = x + pos_embed
        x = self.transformer(x)
        x = self.norm(x)
        cls = x[:, 0]  # (B, d_model)
        return cls


class ArcMarginProduct(nn.Module):
    """ArcFace (ArcMargin) layer producing logits.
    Implementation adapted for stability.
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.3, easy_margin=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.easy_margin = easy_margin

    def forward(self, input, labels=None):
        """Compute ArcFace logits.

        - Training (label-conditioned): pass `labels` to apply angular margin.
        - Inference/validation (label-free): omit `labels` to get cosine logits.
        """
        # Force fp32: ArcFace cos/sin/sqrt are numerically unstable in fp16
        with torch.amp.autocast(device_type="cuda", enabled=False):
            input = input.float()
            cosine = F.linear(F.normalize(input), F.normalize(self.weight.float()))

            if labels is None:
                return cosine * self.s

            sine = torch.sqrt(torch.clamp(1.0 - torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7) ** 2, min=1e-6))
            phi = cosine * self.cos_m - sine * self.sin_m
            if self.easy_margin:
                phi = torch.where(cosine > 0, phi, cosine)
            else:
                phi = torch.where(cosine > self.th, phi, cosine - self.mm)

            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, labels.view(-1, 1), 1.0)

            logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            logits *= self.s
            return logits


class HybridModel(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone=None,
        d_model=512,
        transformer_layers=4,
        backbone_channels=2048,
        input_size=128,
    ):
        super().__init__()
        if backbone is None:
            self.backbone = ResNetBackbone()
        else:
            self.backbone = backbone
        self.head = TransformerHead(
            in_channels=backbone_channels,
            d_model=d_model,
            num_layers=transformer_layers,
            input_size=input_size,
        )
        self.embedding_fc = nn.Sequential(
            nn.Linear(d_model, d_model), nn.BatchNorm1d(d_model)
        )
        self.arcface = ArcMarginProduct(d_model, num_classes, s=30.0, m=0.3)

    def forward(self, x, labels=None):
        featmap = self.backbone(x)
        emb = self.head(featmap)
        # Linear in current precision, BatchNorm in fp32 to prevent
        # running_mean/running_var drift that causes NaN after ~8 epochs under AMP
        emb = self.embedding_fc[0](emb)
        with torch.amp.autocast(device_type="cuda", enabled=False):
            emb = self.embedding_fc[1](emb.float())
        if labels is None:
            return emb
        logits = self.arcface(emb, labels)
        return logits
