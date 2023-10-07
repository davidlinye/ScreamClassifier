from torch import nn
import torch
from torchsummary import summary

class CNNNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # 1 convolutional block + flatten layer + linear layer + softmax
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels = 1,
                out_channels = 16,
                kernel_size = 3,
                stride = 1, 
                padding = 2
                ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2)
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(16*33*44, 6) #Input is size of flatten, output is num of classes
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.flatten(x)
        logits = self.linear(x)
        predictions = self.softmax(logits)

        return predictions

if __name__ == '__main__':
    if torch.cuda.is_available():
        DEVICE = 'cuda'
    else:
        DEVICE = 'cpu'
    cnn = CNNNetwork().to(DEVICE)
    summary(cnn, (1,64,87))      