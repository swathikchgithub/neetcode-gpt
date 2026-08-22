import torch
import torch.nn as nn
from torchtyping import TensorType


class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        # Architecture: Linear(784, 512) -> ReLU -> Dropout(0.2) -> Linear(512, 10) -> Sigmoid
        
        self.fc1 = nn.Linear(784, 512)  # linear layer

        self.relu = nn.ReLU()  # relu activation
        
        self.dropout = nn.Dropout(0.2)  # dropout regularization
        
        self.fc2 = nn.Linear(512, 10)  # output layer
        
        self.sigmoid = nn.Sigmoid()  # sigmoid activation

    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        # images shape: (batch_size, 784)
        # Return the model's prediction to 4 decimal places
        
        out = self.fc1(images)
        
        out = self.relu(out)
        
        out = self.dropout(out)
        
        out = self.fc2(out)
        
        out = self.sigmoid(out)

        return out