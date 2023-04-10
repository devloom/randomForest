import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset, Image
from skimage.transform import resize

#ds = load_dataset("keremberke/pokemon-classification", name="full", split="train[100:200]")


class Dataset:
	def __init__(self, pth = "keremberke/pokemon-classification"):
		self.dataset_path = pth
		self.ds = load_dataset(self.dataset_path, name="full")
		self.train_dataset = self.ds['train']
		self.train_x = self.train_dataset['image']
		self.train_y = self.train_dataset['labels']

	def imgNumpy(self,i):
		img_x = np.asarray(self.train_x[i], dtype=float) / 255
		return img_x

	def train_statistics(self):
		x_mean = np.zeros((224,224,3))
		for i in range(len(self.train_x)):
			x_mean = np.add(x_mean,np.asarray(self.train_x[i], dtype=float) / 255) 

		return np.divide(x_mean,len(self.train_x))


	'''
	def loadData():
		ds = load_dataset("keremberke/pokemon-classification", name="full")
		train = ds['train']
		#train_images = []
		train_images = train['image']
		train_labels = train['labels']
		return train_images, train_labels
		#return 0,0
	'''
	


if __name__ == '__main__':
	dataset = Dataset()
	#img = dataset.imgNumpy(0)
	print(dataset.train_statistics())

	#x_train, y_train = loadData()
	#plt.imshow(x_train[4868])
	#plt.plot()
