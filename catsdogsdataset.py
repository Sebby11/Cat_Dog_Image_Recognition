"""
catsdogsdataset.py
By: Sebastian Medina

Summary:
Class for construction of dataset containing images of cats & dogs.

Pieces of code taken from Aladdin Persson: https://www.youtube.com/watch?v=ZoZHd0Zm3RY
"""

import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from skimage import io

class CatsDogsDataset(Dataset):
	def __init__(self, csv, image_dir, transform=None):
		self.file_names = pd.read_csv(csv)
		self.image_dir = image_dir
		self.transform = transform

	def __len__(self):
		return len(self.file_names)	# 20,000: 10k cats, 10k dogs

	def __getitem__(self, ind):
		"""
		Return specific image & label to image
		"""
		# row ind column column 0: [i, 0]
		img_path = os.path.join(self.image_dir, self.file_names.iloc[ind, 0])
		image = io.imread(img_path)
		y_label = torch.tensor(int(self.file_names.iloc[ind, 1]))

		if self.transform:
			image = self.transform(image)

		return image, y_label