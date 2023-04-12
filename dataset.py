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
		#self.train_x = self.train_dataset['image']
		self.pixels = 56
		self.train_img = [image.convert("RGB").resize((self.pixels,self.pixels)) for image in self.train_dataset["image"]]
		self.train_x = np.array([self.imgNumpy(image) for image in self.train_img])
		self.train_y = np.array(self.train_dataset['labels'])

	def imgNumpy(self,img):
		img_x = np.asarray(img, dtype=float) / 255
		return img_x

	def train_statistics(self):
		x_mean = np.zeros((self.pixels,self.pixels,3))
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
	print(dataset.train_x)
	print(dataset.train_y)
	#img = dataset.imgNumpy(0)
	#print(img)
	#print(dataset.train_statistics())

	#x_train, y_train = loadData()
	#plt.imshow(dataset.train_x[4868])
	#plt.show()
