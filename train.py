"""
train.py
By: Sebastian Medina

Summary:
Set to be main file for training the network

Lines 26-27 from Aladdin Persson w/ minor changes: https://www.youtube.com/watch?v=ZoZHd0Zm3RY
"""

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision.transforms as transforms

import matplotlib.pyplot as plt

from PIL import Image

from catsdogsdataset import CatsDogsDataset

# 1.) Get Data & place it to batch load w/ DataLoader
training_data = CatsDogsDataset(csv='cats_dogs.csv', image_dir='data\\train\\both', 
								transform=transforms.ToTensor())

train_loader = DataLoader(dataset=training_data, batch_size=100, shuffle=True)

'''
# Test print an image from the data
images, labels = next(iter(train_loader))

print("Images: \n", images, "\n\n")
print("Labels: ", labels, "\n\n")
plt.imshow(images[2].permute(1, 2, 0))
plt.show()
'''

# 2.) Get model // ResNet
class ResNet(nn.Module):
	def __init__(self):
		super().__init__()
		self.layer1 = nn.Linear(250 * 250, 64)
		self.layer2 = nn.Linear(64, 64)
		self.layer3 = nn.Linear(64, 10)
		self.do     = nn.Dropout(0.1)

	def forward(self, x):
		hiddel_Layer1 = nn.functional.relu(self.layer1(x))
		hiddel_Layer2 = nn.functional.relu(self.layer2(h1))
		dropout       = self.do(h2 + h1)
		prediction	  = self.layer3(dropout)
		return prediction
model = ResNet()

# 3.) Get optimizer
optimizer = optim.SGD(model.parameters(), lr=1e-2)

# 4.) Get loss(objective) function
loss_function = nn.CrossEntropyLoss()


# 5.) Training Loop
for epoch in range(100):
	losses = list()

	for batch in train_loader:
		inputImage, label = batch
		print("X: ", inputImage.size())
		print("Y: ", label)
		exit()