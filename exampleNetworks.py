import torch
import torch.nn as nn
import torch.nn.functional as F

# Model
class Net(nn.Module):
    
    def __init__(self, weight):
        super(Net, self).__init__()
        # Initializes the weights of the convolutional layer 
        # to be the weights of the 4 defined filters
        k_height, k_width = weight.shape[2:]

        # Assumes there are 4 grayscale filters
        self.conv = nn.Conv2d(in_channels=1, out_channels=4, 
                                kernel_size=(k_height, k_width), bias=False)

        self.conv.weight = torch.nn.Parameter(weight)


    def forward(self, x):
        # Calculates the output of a convolutional layer pre- and post-activation
        conv_x = self.conv(x)
        activated_x = F.relu(conv_x)
            
        # Returns both layers
        return conv_x, activated_x