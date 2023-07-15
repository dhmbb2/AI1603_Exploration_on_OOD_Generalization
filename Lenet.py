import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LeNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2) -> None:
        super().__init__()
        self.num_classes = out_channels
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.num_classes)
        for i in [self.conv1, self.conv2, self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_normal_(i.weight)
            nn.init.zeros_(i.bias)

    def forward(self, X):
        out = self.bn1(self.pool1(F.relu(self.conv1(X))))
        out = self.bn2(self.pool2(F.relu(self.conv2(out))))
        out = out.view(-1, 16*4*4)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

class LeNet_NoBN(nn.Module):
    def __init__(self, in_channels=3, out_channels=2) -> None:
        super().__init__()
        self.num_classes = out_channels
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.num_classes)
        for i in [self.conv1, self.conv2, self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_normal_(i.weight)
            nn.init.zeros_(i.bias)

    def forward(self, X):
        out = self.pool1(F.relu(self.conv1(X)))
        out = self.pool2(F.relu(self.conv2(out)))
        out = out.view(-1, 16*4*4)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out