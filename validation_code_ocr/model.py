import torch
import torch.nn as nn
import torch.nn.functional as F


class CRNN(nn.Module):
    def __init__(self, input_channels=3, hidden_size=64, num_classes=11):
        super(CRNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 8, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Recurrent layers
        self.rnn = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        # Fully connected layer
        self.fc1 = nn.Linear(429, 128)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # Input shape: (batch_size, channels, height, width)
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        # x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        # x = self.pool(x)
        x = F.relu(self.conv3(x))
        # x = self.pool(x)

        # Reshape for RNN
        x = nn.Flatten(2)(x)
        x = x = F.relu(self.fc1(x))
        # RNN
        x, _ = self.rnn(x)
        # Fully connected layer
        x = self.fc(x)  # Take the output from the last time step
        return x
