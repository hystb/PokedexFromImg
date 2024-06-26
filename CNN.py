import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data

class CNN(nn.Module):
    def __init__(self, maxId):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, stride=2)
        
        self.fc1 = nn.Linear(in_features=128 * 16 * 16, out_features=256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(in_features=128, out_features=maxId)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool2(x)
        
        x = torch.relu(self.bn4(self.conv4(x)))
        x = self.pool3(x)
        
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
