"""
csvCreator.py
By: Sebastian Medina

Summary:
Grabs names of image data for training and places them in a 
'.csv' file with the appropriate label. Also resizes the image
if it's not to your liking.
"""

import csv
import random
import torchvision.transforms as transforms
from os import listdir
from os.path import isfile, join
from PIL import Image

# Clear & start w/ empty csv so you don't keep buildling on one
csvName = 'cats_dogs.csv'
f = open(csvName, 'w')
f.truncate()
f.close()

# Grab file names
#catfiles = [f for f in listdir('data\\train\\cats') if isfile(join('data\\train\\cats', f))]
#dogfiles = [f for f in listdir('data\\train\\dogs') if isfile(join('data\\train\\dogs', f))]
overall = [f for f in listdir('data\\train\\both') if isfile(join('data\\train\\both', f))]

# Start csv writer
f = open(csvName, 'w', newline='')
writer = csv.writer(f)

# Resizing pipeline
p = transforms.Compose([transforms.Resize((250, 250))])

# Writing/Resizing loop
for fname in overall:
	
	if "cat" in fname:
		writer.writerow([fname, 1])

		'''
		# Uncomment if resizing images
		name = 'data\\train\\both\\'+fname
		img = Image.open(name)
		img = p(img)
		img.save(name)
		'''
	else:
		writer.writerow([fname, 0])

		'''
		# Uncomment if resizing images
		name = 'data\\train\\dogs\\'+fname
		img = Image.open(name)
		img = p(img)
		img.save(name)
		'''
		
f.close()