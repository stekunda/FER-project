import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary

import pandas as pd
import numpy as np


class FERModel(nn.Module):
    def __init__(self, num_classes):
        super(FERModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout2d(p=0.5)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout2d(p=0.5)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(64 * 12 * 12, 256)
        self.drop3 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.drop1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.drop2(x)

        x = self.flatten(x)

        x = x.view(-1, 64 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = self.drop3(x)
        predictions = F.softmax(self.fc2(x), dim=1)

        return predictions


if __name__ == '__main__':
    cnn = FERModel(7)
    summary(cnn, (1, 48, 48))
