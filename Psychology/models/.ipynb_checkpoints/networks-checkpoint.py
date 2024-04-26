import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class CNN(nn.Module):
    def __init__(self, input_channels=3):
        super(CNN, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=5, padding=2) # 224→112
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2) # 112→ 56
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, padding=2) # 56→28
        self.fc1 = nn.Linear(64 * 28 * 28, 512)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 64 * 28 * 28)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(224 * 224 * 3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, x):
        x = x.view(-1, 224 * 224 * 3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ResNetRegression(nn.Module):
    def __init__(self, num_features=1):
        super(ResNetRegression, self).__init__()
        self.resnet = torchvision.models.resnet18(weights=torchvision.models.resnet.ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(in_features=self.resnet.fc.in_features, out_features=num_features)
    def forward(self, x):
        return self.resnet(x)

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.vgg16 = torchvision.models.vgg16(weights=torchvision.models.vgg.VGG16_Weights.DEFAULT)
        num_features = self.vgg16.classifier[-1].in_features
        self.vgg16.classifier[-1] = nn.Linear(num_features, 1)
    def forward(self, x):
        return self.vgg16(x)
        
class VIT(nn.Module):
    def __init__(self, num_classes=1):
        super(VIT, self).__init__()
        self.vit = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.DEFAULT)
        self.vit.heads = nn.Linear(self.vit.heads[0].in_features, num_classes)
    def forward(self, x):
        return self.vit(x)

if __name__ == "__main__":
    model = CNN(input_channels=3)
    print(model)
    input = torch.randn(4, 3, 224, 224)
    output = model(input)
    print(output)
