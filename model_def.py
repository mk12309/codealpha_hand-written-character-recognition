import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        # Input: 1 x 28 x 28
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Output: 32 x 28 x 28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Output: 64 x 28 x 28
        self.pool1 = nn.MaxPool2d(2, 2)  # Output: 64 x 14 x 14
        self.dropout1 = nn.Dropout2d(0.25)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Output: 128 x 14 x 14
        self.pool2 = nn.MaxPool2d(2, 2)  # Output: 128 x 7 x 7
        self.dropout2 = nn.Dropout2d(0.25)
        
        self.fc1 = nn.Linear(128 * 7 * 7, 128)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, 1, 28, 28)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = x.view(-1, 128 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        return x

def preprocess_images(images):
    """
    Preprocess raw numpy images (either flat 784, 28x28, or 28x28x1) to (N, 1, 28, 28)
    and normalize values to range [0.0, 1.0].
    """
    images = images.astype('float32')
    
    # Normalize if data is in range [0, 255]
    if images.max() > 1.0:
        images /= 255.0
        
    # Check dimensionality
    if images.ndim == 2:
        # Flattened (N, 784)
        N = images.shape[0]
        images = images.reshape(N, 1, 28, 28)
    elif images.ndim == 3:
        # (N, 28, 28)
        N = images.shape[0]
        images = np.expand_dims(images, axis=1)
    elif images.ndim == 4:
        # (N, 28, 28, 1) or (N, 1, 28, 28)
        if images.shape[1] == 1:
            pass  # Already (N, 1, 28, 28)
        elif images.shape[3] == 1:
            images = np.transpose(images, (0, 3, 1, 2))
        else:
            raise ValueError(f"Unexpected image shape: {images.shape}")
            
    return images
