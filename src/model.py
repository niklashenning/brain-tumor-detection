import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        # Create two convolutional layers that transform the input from
        # 1 channel to 64 channels, a max pooling layer and two
        # fully connected layers with a dropout layer to reduce overfitting.
        # The second fully connected layer has an output of size 4
        # (the amount of classes in the dataset)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 64 * 64, 256)
        self.fc2 = nn.Linear(256, 4)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # Pass the input through the convolutional layers with ReLU
        # activation and max pooling before flattening the input images,
        # applying dropout, and finally passing it through the
        # fully connected layers and ReLU activation function
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 64 * 64)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
