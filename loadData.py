import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset, Image

#ds = load_dataset("keremberke/pokemon-classification", name="full", split="train[100:200]")

def loadData():
	ds = load_dataset("keremberke/pokemon-classification", name="full")
	train = ds['train']
	#train_images = []
	train_images = train['image']
	train_labels = train['labels']
	'''
	for i in range(len(train)):
		if (i%1000 == 0):
			print(i)
		img_data = train[i]['image'].getdata()

		img_as_list = np.asarray(img_data, dtype=float) / 255
		#img_as_list = img_as_list.reshape(img.size)

		#train_images.append(img_as_list)

		#print(img_as_list)
	'''
	return train_images, train_labels
	#return 0,0

def dataNumpy(img):
	img_data = img.getdata()
	img_as_list = np.asarray(img_data, dtype=float) / 255
	return img_as_list

if __name__ == '__main__':
	x_train, y_train = loadData()
	#plt.imshow(x_train[4868])
	#plt.plot()
