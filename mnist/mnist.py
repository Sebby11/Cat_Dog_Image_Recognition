import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim

import torchvision
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader

# Data
train_data = datasets.MNIST('../data', train=True, download=True, transform=transforms.ToTensor())
train, val = torch.utils.data.random_split(train_data, [55000, 5000])		# Split to training data / validation data
train_loader = DataLoader(train, batch_size=32)
val_loader= DataLoader(val, batch_size=32)

images, labels = next(iter(val_loader))
#print("Images: \n", images, "\n\n")
#print("Labels: ", labels, "\n\n")
#plt.imshow(images[2].reshape(28, 28), cmap="gray")
#plt.show()

# Network
"""
model = nn.Sequential(
			nn.Linear(28 * 28, 64),	# Mnist images 28x28 so each pixel as input -> 64 dim layer
			nn.ReLU(),				# 64 node hidden layer
			nn.Linear(64, 64),
			nn.ReLU(),
			nn.Dropout(0.1)			# if we're overfitting
			nn.Linear(64, 10)		# 10 digits: 0-9
		).cuda()
"""
class ResNet(nn.Module):
	def __init__(self):
		super().__init__()
		self.l1 = nn.Linear(28 * 28, 64)
		self.l2 = nn.Linear(64, 64)
		self.l3 = nn.Linear(64, 10)
		self.do = nn.Dropout(0.1)

	def forward(self, x):
		print("X: ", x.size(), "\n", x)
		exit()
		h1 = nn.functional.relu(self.l1(x))
		h2 = nn.functional.relu(self.l2(h1))
		do = self.do(h2 + h1)			# if h1 or h2 are 0 they get dropped out
		prediction = self.l3(do)
		return prediction

model = ResNet().cuda()


# Optimizer
optimizer = optim.SGD(model.parameters(), lr=1e-2)

# Loss Function
loss_Function = nn.CrossEntropyLoss()


# Training & Validation Loop
numEpochs = 5
for epoch in range(numEpochs):
	losses = list()

	# Training Data Loader
	for batch in train_loader:
		# +Image (x) must be a long vector to go into a model.
		# 		-(x) is image data
		# +Label (y)
		x, y = batch

		# x: (batch size) * 1 * 28 * 28
		#		- Since it's a b&w image instead of 3 layers it's just 1
		#		- x is a long tensor of image data
		b = x.size(0)
		x = x.view(b, -1).cuda()			# x data --> b x 784 OR (b rows x (28*28) columns)

		#Network Training Steps

		# 1: Forward propogation / going through NN
		prediction = model(x)


		# 2: Compute loss (or objective) function
		loss = loss_Function(prediction, y.cuda())

		# 3: Stop accumulation of gradients 
		#		- (can use model or optimizer since both contain model parameters)
		model.zero_grad()

		# 4: (Backpropogation) Computer partial derivatives of the loss w/ respect to parameters
		loss.backward()

		# 5: (Gradient Descent) Step in opposite direction of the gradient
		optimizer.step()

		# 6: get loss avg to show progress later
		losses.append(loss.item())

	# 7: Show Progress (Create tensor of losses, then compute mean of losses tensor)
	print(f'Epoch {epoch + 1}: Training loss: {torch.tensor(losses).mean():.2f}')

	# Validation
	losses = list()
	for batch in train_loader:
		x, y = batch

		b = x.size(0)
		x = x.view(b, -1).cuda()			# x data --> b x 28 x 28 OR (b rows x (28*28) columns)

		# 1: Forward propogation / going through NN
		with torch.no_grad():
			prediction = model(x)


		# 2: Compute loss (or objective) function
		loss = loss_Function(prediction, y.cuda())

		losses.append(loss.item())

	# Print validation loss
	print(f'Epoch {epoch + 1}: Validation loss: {torch.tensor(losses).mean():.2f}')

#print("Prediction: ", prediction)

# Testing with photos
for i in range(5):
	b = images[i].size(0)
	p = images[i].view(b, -1).cuda()
	with torch.no_grad():
		prediction = model(p)
	prediction = prediction.tolist()
	print(prediction)
	print("Value predicted to be: ", prediction[0].index(max(prediction[0])))
	plt.imshow(images[i].reshape(28, 28), cmap="gray")
	plt.show()