import torch
import torch.nn as nn
import torchvision.models as models

def get_mobilenet_v2(num_classes=10, pretrained=False):
    """
    Model B: MobileNetV2 (Arsitektur Standar Industri Efisien)
    Diadaptasi untuk citra input grayscale (1 kanal) 28x28.
    Memiliki 2.2M parameter dengan blok Inverted Residual & Depthwise Separable Conv.
    """
    if pretrained:
        try:
            weights = models.MobileNet_V2_Weights.DEFAULT
            model = models.mobilenet_v2(weights=weights)
        except Exception:
            model = models.mobilenet_v2(weights=None)
    else:
        model = models.mobilenet_v2(weights=None)

    # Sesuaikan conv pertama: in_channels=1, stride=1 (agar resolusi 28x28 optimal)
    model.features[0][0] = nn.Conv2d(
        in_channels=1,
        out_channels=32,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )

    # Sesuaikan classifier output ke 10 kelas
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model

if __name__ == "__main__":
    model = get_mobilenet_v2()
    dummy_input = torch.randn(4, 1, 28, 28)
    output = model(dummy_input)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"MobileNetV2 berhasil diinisialisasi.")
    print(f"Total parameter: {total_params:,}")
    print(f"Bentuk output: {output.shape}")
