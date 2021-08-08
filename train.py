"""
train.py
By: Sebastian Medina

Summary:
Set to be main file for training the network

Lines 26-27 from Aladdin Persson w/ minor changes: https://www.youtube.com/watch?v=ZoZHd0Zm3RY
"""

import numpy as np
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
				transforms.Resize((150, 150))
			])
# 1.) Get Data & place it to batch load w/ DataLoader
training_data = CatsDogsDataset(csv='cats_dogs.csv', image_dir='data\\train\\both', 
								transform=transform)

#train, val = torch.utils.data.random_split(training_data, [5000, 15000])

train_loader = DataLoader(dataset=training_data, batch_size=100, shuffle=True)

'''
# Test print images
for image, label in train_loader:
	fig, ax = plt.subplots(figsize=(16, 12))
	ax.set_xticks([])
	ax.set_yticks([])
	ax.imshow(torchvision.utils.make_grid(image, nrow=8).permute(1, 2, 0))
	plt.show()
	break
'''



# 2.) Get model // ResNet
class network(nn.Module):
	def __init__(self):
		super().__init__()
		self.model = nn.Sequential(
				nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, padding=1),
				nn.ReLU(),
				nn.Conv2d(32, 64, kernel_size=3, padding=1),
				nn.ReLU(),
				nn.MaxPool2d(2,2),

				nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
				nn.ReLU(),
				nn.Conv2d(128, 128, kernel_size=3, padding=1),
				nn.ReLU(),
				nn.MaxPool2d(2,2),

				nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
				nn.ReLU(),
				nn.Conv2d(256, 256, kernel_size=3, padding=1),
				nn.ReLU(),
				nn.MaxPool2d(2,2),

				nn.Flatten(),
				nn.Linear(82944, 1024),
				nn.ReLU(),
				nn.Linear(1024, 512),
				nn.ReLU(),
				nn.Linear(512, 1)
			)

	def forward(self, x):
		return nn.functional.relu(self.model(x))

model = network()


# 3.) Get optimizer
optimizer = optim.SGD(model.parameters(), lr=.001, momentum=0.9)

# 4.) Get loss(objective) function
loss_function = nn.BCELoss()


# 5.) Training Loop
for epoch in range(2):
	total_loss = 0.0
	cnt = 1
	for batch in train_loader:
		inputImages, labels = batch

		optimizer.zero_grad()

		labels = labels.view(100, 1)
		labels = labels.float()
		
		prediction = model(inputImages)
		#plt.imshow(inputImages[0].permute(1, 2, 0))
		#plt.show()

		loss = loss_function(prediction, labels)
		print("Loss {}: {}".format(cnt, loss))
		cnt += 1
		print("Correct Label: ", labels[0])
		print("Predicted label: ", prediction[0])
		print("\n\n")

		loss.backward()

		optimizer.step()

		total_loss += loss.item()

	print("Epoch {}: Loss {}".format(epoch, total_loss))


# 6.) Testing
images, labels = next(iter(train_loader))

print("Images: \n", images[2], "\n\n")
print("Labels: ", labels[2], "\n\n")

prediction = model(images)
print("PREDICTION: ", prediction)

plt.imshow(images[2].permute(1, 2, 0))
plt.show()

PATH = './my_net.pth'
torch.save(model.state_dict(), PATH)