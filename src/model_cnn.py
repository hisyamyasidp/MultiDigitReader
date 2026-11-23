import torch
import torch.nn as nn

class CustomDigitCNN(nn.Module):
    """
    Model A: Arsitektur Custom Convolutional Neural Network (CNN)
    Dirancang khusus untuk klasifikasi digit tulisan tangan 28x28.
    Memiliki parameter yang ringkas, cepat di CPU, dan tahan overfitting.
    """
    def __init__(self, num_classes=10):
        super(CustomDigitCNN, self).__init__()
        
        # Blok Konvolusi 1
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 14x14
            nn.Dropout2d(0.25)
        )
        
        # Blok Konvolusi 2
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 7x7
            nn.Dropout2d(0.25)
        )
        
        # Lapisan Fully Connected (Dense)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        logits = self.classifier(x)
        return logits

def get_custom_cnn(num_classes=10):
    return CustomDigitCNN(num_classes=num_classes)

if __name__ == '__main__':
    model = get_custom_cnn()
    dummy_input = torch.randn(4, 1, 28, 28)
    output = model(dummy_input)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Custom CNN berhasil diinisialisasi.")
    print(f"Total parameter: {total_params:,}")
    print(f"Bentuk output: {output.shape}")
