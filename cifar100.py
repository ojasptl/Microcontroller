import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

# Define the Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# Define the ResNet model with Multi-Head Attention
class ResNetWithAttention(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNetWithAttention, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # ResNet blocks
        self.layers = nn.Sequential(
            ResidualBlock(64, 64),
            nn.MultiheadAttention(embed_dim=64, num_heads=8, batch_first=True),
            ResidualBlock(64, 128, stride=2),
            nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True),
            ResidualBlock(128, 256, stride=2),
            nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True),
            ResidualBlock(256, 512, stride=2),
            nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True),
        )

        self.fc = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))

        # Apply ResNet blocks with attention
        for i in range(len(self.layers) // 2):
            res_block = self.layers[i * 2]
            attn = self.layers[i * 2 + 1]

            out = res_block(out)
            out_flattened = out.view(out.size(0), -1, out.size(1))  # Prepare for attention
            attn_out, _ = attn(out_flattened, out_flattened, out_flattened)
            out = out + attn_out.reshape_as(out)  # Add residual connection

        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

# Training and Evaluation Function
def train_and_evaluate():
    # Hyperparameters
    num_epochs = 10
    batch_size = 32
    learning_rate = 0.001

    # CIFAR-100 Dataset
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))
    ])

    train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Model, Loss, Optimizer
    device = torch.device('cpu')  # Use CPU
    model = ResNetWithAttention(num_classes=100).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        correct = 0
        total = 0
        running_loss = 0.0

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # Update progress bar with training accuracy
                pbar.set_postfix(
                    Loss=f"{running_loss / (i + 1):.4f}",
                    Accuracy=f"{100 * correct / total:.2f}%"
                )
                pbar.update(1)

            print(f"Epoch {epoch + 1}/{num_epochs} completed. Training Accuracy: {100 * correct / total:.2f}%")

        # Testing
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        print(f"Epoch {epoch + 1}/{num_epochs} completed. Test Accuracy: {100 * correct / total:.2f}%")
        torch.save(model, "model.pth")

if __name__ == "__main__":
    train_and_evaluate()

