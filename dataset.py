import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset, Image

#ds = load_dataset("keremberke/pokemon-classification", name="full", split="train[100:200]")
#"keremberke/pokemon-classification",
#"Bingsu/Cat_and_Dog"
#"jbarat/plant_species"
#"cifar10"

class Dataset:
	def __init__(self, pth = "cifar10"):
		self.dataset_path = pth
		self.ds = load_dataset(self.dataset_path)
		self.pixels = 32
		self.train_dataset = self.ds['train']

		self.test_dataset = self.ds['test']

		self.test_img = [image.convert("RGB").resize((self.pixels,self.pixels)) for image in self.test_dataset["img"]]
		self.test_x = np.array([self.imgNumpy(image) for image in self.test_img])
		self.test_y = np.array(self.test_dataset['label'])

	def imgNumpy(self,img):
		img_x = np.asarray(img, dtype=float) / 255
		return img_x

	def train_statistics(self):
		x_mean = np.zeros((self.pixels,self.pixels,3))
		cat_mean = np.zeros((self.pixels,self.pixels,3))
		dog_mean = np.zeros((self.pixels,self.pixels,3))
		for i in range(len(self.test_x)):
			x_mean = np.add(x_mean,np.asarray(self.test_x[i], dtype=float) / 255) 
			if self.test_y[i] == 0:
				cat_mean = np.add(cat_mean,np.asarray(self.test_x[i], dtype=float))
			if self.test_y[i] == 1:
				dog_mean = np.add(dog_mean,np.asarray(self.test_x[i], dtype=float))
                

		fig, ax = plt.subplots(nrows=1,ncols=2)
		ax[0].imshow(dog_mean/1000)
		ax[1].imshow(cat_mean/1000)
		fig, axs = plt.subplots(nrows=5, ncols=5)
		axs = axs.flat
		for i in range(25):
			idx = np.random.randint(1000,2000)
			axs[i].imshow(self.test_x[idx])  
		plt.show()

		return np.divide(x_mean,len(self.test_x))
	


if __name__ == '__main__':
	dataset = Dataset()
	print(dataset.train_img)
