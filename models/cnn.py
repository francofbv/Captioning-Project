import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        '''
        Model architecture: CNN for frame embedding generation
        Input: 224x224x3
        Output: 2048-dimensional frame embedding
        '''
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.bn5 = nn.BatchNorm2d(1024)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the size of flattened features
        # After 5 pooling layers: 224 -> 112 -> 56 -> 28 -> 14 -> 7
        self.fc = nn.Linear(1024 * 7 * 7, 2048)
        
    def forward(self, x):
        # First conv block
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Second conv block
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Third conv block
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Fourth conv block
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Fifth conv block
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        
        # Flatten the features
        x = x.view(-1, 1024 * 7 * 7)
        
        # Generate frame embedding
        x = F.relu(self.fc(x))
        
        return x
        
        
        