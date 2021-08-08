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

transform = transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize([0.5,0.5,0.5],
									[0.5,0.5,0.5])
			])
# 1.) Get Data & place it to batch load w/ DataLoader
training_data = CatsDogsDataset(csv='cats_dogs.csv', image_dir='data\\train\\both', 
								transform=transform)

train, val = torch.utils.data.random_split(training_data, [1000, 19000])

train_loader = DataLoader(dataset=train, batch_size=10, shuffle=True)

'''
# Test print an image from the data
images, labels = next(iter(train_loader))

print("Images: \n", images, "\n\n")
print("Labels: ", labels, "\n\n")
for i in range(10):
	plt.imshow(images[i].permute(1, 2, 0))
	plt.show()
exit()
'''


# 2.) Get model // ResNet
class network(nn.Module):
	def __init__(self):
		super().__init__()
		self.l1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
		self.l2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
		self.l3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5)

		self.maxPooling = nn.MaxPool2d(kernel_size=4)
		self.adaptive = nn.AdaptiveAvgPool1d(256)

		self.l4 = nn.Linear(in_features=256, out_features=128)
		self.l5 = nn.Linear(in_features=128, out_features=64)
		self.out = nn.Linear(in_features=64, out_features=1)

	def forward(self, x):
		x = self.l1(x)
		#max pooling layer
		x = self.maxPooling(x)
		x = nn.functional.relu(x)
		
		x = self.l2(x)
		x = self.maxPooling(x)
		x = nn.functional.relu(x)

		x = self.l3(x)
		x = self.maxPooling(x)
		x = nn.functional.relu(x)

		x = nn.functional.dropout(x)
		x = x.view(1, x.size()[0], -1)
		x = self.adaptive(x).squeeze()

		x = self.l4(x)
		x = nn.functional.relu(x)

		x = self.l5(x)
		x = nn.functional.relu(x)

		x = nn.functional.relu(self.out(x))

		return x
	
model = network()


# 3.) Get optimizer
optimizer = optim.SGD(model.parameters(), lr=1e-2)

# 4.) Get loss(objective) function
loss_function = nn.BCELoss()


# 5.) Training Loop
for epoch in range(2):
	total_loss = 0.0
	cnt = 1
	for batch in train_loader:
		inputImages, labels = batch

		labels = labels.view(10, 1)
		labels = labels.float()
		#inputImages = inputImages.view(inputImages.size(0), -1)
		#print("INPUT IMAGES: ", inputImages.size())
		
		
		prediction = model(inputImages)
		#exit()

		loss = loss_function(prediction, labels)
		#print(prediction)
		#print(labels)
		#print()
		print("Loss {}: {}".format(cnt, loss))
		cnt += 1
		#exit()

		optimizer.zero_grad()

		loss.backward()

		optimizer.step()

		total_loss += loss.item()

	print("Epoch {}: Loss {}".format(epoch, total_loss))

images, labels = next(iter(train_loader))

print("Images: \n", images[2], "\n\n")
print("Labels: ", labels[2], "\n\n")

prediction = model(images)
print("PREDICTION: ", prediction)

plt.imshow(images[2].permute(1, 2, 0))
plt.show()