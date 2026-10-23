import torch
import torch.nn as nn
import torchvision.models as models

class VisualStream(nn.Module):
    def __init__(self, pretrained=True):
        super(VisualStream, self).__init__()
        weights = models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b4(weights=weights)
        # Extract features before the final classifier (1792 channels)
        self.feature_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

    def forward(self, x):
        return self.backbone(x)  # Output shape: (batch_size, 1792)


class AudioStream(nn.Module):
    def __init__(self, pretrained=True):
        super(AudioStream, self).__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        
        # Adapt first conv layer for 1-channel Mel-Spectrogram input
        original_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias
        )
        
        # Copy RGB weights averaged across 3 channels to 1 channel
        if pretrained:
            with torch.no_grad():
                self.backbone.conv1.weight = nn.Parameter(
                    original_conv.weight.mean(dim=1, keepdim=True)
                )

        # Extract features before final FC (512 channels)
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        return self.backbone(x)  # Output shape: (batch_size, 512)


class CrossAttentionFusion(nn.Module):
    def __init__(self, visual_dim=1792, audio_dim=512, embed_dim=256, num_heads=4):
        super(CrossAttentionFusion, self).__init__()
        self.proj_visual = nn.Linear(visual_dim, embed_dim)
        self.proj_audio = nn.Linear(audio_dim, embed_dim)
        
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, visual_feat, audio_feat):
        # Project both vectors to shared embedding space
        v_proj = self.proj_visual(visual_feat).unsqueeze(1)  # (batch, 1, embed_dim)
        a_proj = self.proj_audio(audio_feat).unsqueeze(1)    # (batch, 1, embed_dim)
        
        # Visual acts as Query, Audio acts as Key/Value
        attn_out, _ = self.multihead_attn(query=v_proj, key=a_proj, value=a_proj)
        fused = self.layer_norm(v_proj + attn_out).squeeze(1) # Residual connection
        
        return fused  # Output shape: (batch_size, 256)


class MultimodalDeepfakeDetector(nn.Module):
    def __init__(self, pretrained=True):
        super(MultimodalDeepfakeDetector, self).__init__()
        self.visual_stream = VisualStream(pretrained=pretrained)
        self.audio_stream = AudioStream(pretrained=pretrained)
        self.fusion = CrossAttentionFusion(
            visual_dim=self.visual_stream.feature_dim,
            audio_dim=self.audio_stream.feature_dim,
            embed_dim=256
        )
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1)  # Binary logit (Real vs Fake)
        )

    def forward(self, face_tensor, audio_tensor):
        v_feat = self.visual_stream(face_tensor)
        a_feat = self.audio_stream(audio_tensor)
        fused_feat = self.fusion(v_feat, a_feat)
        logit = self.classifier(fused_feat)
        return logit


if __name__ == "__main__":
    print("Testing Multimodal Deepfake Detector Architecture...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Instantiate model
    model = MultimodalDeepfakeDetector(pretrained=True).to(device)
    model.eval()

    # Create dummy batch (batch_size = 2)
    dummy_faces = torch.randn(2, 3, 224, 224).to(device)       # RGB Face crop
    dummy_audio = torch.randn(2, 1, 128, 94).to(device)        # Mel-Spectrogram (128 bins, ~3 sec)

    with torch.no_grad():
        output = model(dummy_faces, dummy_audio)

    print(f"✓ Forward pass successful!")
    print(f"✓ Output tensor shape: {output.shape} (Expected: [2, 1])")
    print(f"✓ Sample Logits: {output.squeeze().cpu().numpy()}")