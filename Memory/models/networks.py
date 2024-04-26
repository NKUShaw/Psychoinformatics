import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from utils.args import get_args

args = get_args()
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
        
class CNN(nn.Module):
    def __init__(self, input_channels):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = x.view(-1, 128 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2), #224→55
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), #55→27
            nn.Conv2d(64, 192, kernel_size=5, padding=2), #27→27
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), #27→13
            nn.Conv2d(192, 384, kernel_size=3, padding=1), #13→13
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), #13→13
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), #13→13
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=3, stride=2), #13→6
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1),
        )
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.vgg16 = torchvision.models.vgg16(weights=torchvision.models.vgg.VGG16_Weights.DEFAULT)
        num_features = self.vgg16.classifier[-1].in_features
        self.vgg16.classifier[-1] = nn.Linear(num_features, 1)
    def forward(self, x):
        return self.vgg16(x)

class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        self.vgg19 = torchvision.models.vgg19(weights=torchvision.models.vgg.VGG19_Weights.DEFAULT)
        num_features = self.vgg19.classifier[-1].in_features
        self.vgg19.classifier[-1] = nn.Linear(num_features, 1)
    def forward(self, x):
        return self.vgg19(x)

class ResNet18(nn.Module):
    def __init__(self, num_features=1):
        super(ResNet18, self).__init__()
        self.resnet = torchvision.models.resnet18(weights=torchvision.models.resnet.ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(in_features=self.resnet.fc.in_features, out_features=num_features)
    def forward(self, x):
        return self.resnet(x)
        
class ResNet50(nn.Module):
    def __init__(self, num_features=1):
        super(ResNet50, self).__init__()
        self.resnet = torchvision.models.resnet50(weights=torchvision.models.resnet.ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(in_features=self.resnet.fc.in_features, out_features=num_features)
    def forward(self, x):
        return self.resnet(x)

class ResNet101(nn.Module):
    def __init__(self, num_features=1):
        super(ResNet101, self).__init__()
        self.resnet = torchvision.models.resnet101(weights=torchvision.models.resnet.ResNet101_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(in_features=self.resnet.fc.in_features, out_features=num_features)
    def forward(self, x):
        return self.resnet(x)

class VIT(nn.Module):
    def __init__(self, num_classes=1):
        super(VIT, self).__init__()
        self.vit = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.DEFAULT)
        self.vit.heads = nn.Linear(self.vit.heads[0].in_features, num_classes)
    def forward(self, x):
        return self.vit(x)

# class AlexNet(nn.Module):
#     def __init__(self, num_classes=1):
#         super(AlexNet, self).__init__()
#         self.alexnet = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.DEFAULT)
#         self.alexnet.classifier[-1] = nn.Linear(4096, num_classes)
#     def forward(self, x):
#         return self.alexnet(x)
        
if __name__ == "__main__":
    model = MLP()
    print(model)
    input = torch.randn(4, 3, 224, 224)
    output = model(input)
    print(output)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {total_params}")
