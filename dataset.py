import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset, Image
#from skimage.transform import resize

#ds = load_dataset("keremberke/pokemon-classification", name="full", split="train[100:200]")
#"keremberke/pokemon-classification",
#"Bingsu/Cat_and_Dog"

class Dataset:
    def __init__(self, pth = "Bingsu/Cat_and_Dog"):
        self.dataset_path = pth
        #self.ds = load_dataset(self.dataset_path, name="full",split="train[2000:6000]")
        self.ds = load_dataset(self.dataset_path, name="full")

        #dataset = load_dataset("parquet", data_files={'train': 'train.parquet', 'test': 'test.parquet'})

        #self.train_dataset = self.ds['train']
        #self.train_dataset = self.ds
        #self.train_dataset = load_dataset(self.dataset_path, name="full",split="train[2000:6000]")
        #self.test_dataset = load_dataset(self.dataset_path, name="full",split="test[:2000]")
        self.pixels = 224
        self.train_dataset = self.ds['train']
        self.test_dataset = self.ds['test']

        self.test_img = [image.convert("RGB").resize((self.pixels,self.pixels)) for image in self.test_dataset["image"]]
        self.test_x = np.array([self.imgNumpy(image) for image in self.test_img])
        self.test_y = np.array(self.test_dataset['labels'])

        #self.train_img = self.train_dataset['image'][3000:3500]
        #self.train_img = [image.convert("RGB").resize((self.pixels,self.pixels)) for image in self.train_dataset["image"][3000:3500]]
        #self.train_x = self.train_dataset['image']
        '''
        self.pixels = 56

        self.train_img = [image.convert("RGB").resize((self.pixels,self.pixels)) for image in self.train_dataset["image"]]
        self.train_x = np.array([self.imgNumpy(image) for image in self.train_img])
        self.train_y = np.array(self.train_dataset['labels'])

        self.test_img = [image.convert("RGB").resize((self.pixels,self.pixels)) for image in self.test_dataset["image"]]
        self.test_x = np.array([self.imgNumpy(image) for image in self.test_img])
        self.test_y = np.array(self.test_dataset['labels'])
        '''

    def imgNumpy(self,img):
        img_x = np.asarray(img, dtype=float) / 255
        return img_x

    def train_statistics(self):
        x_mean = np.zeros((self.pixels,self.pixels,3))
        cat_mean = np.zeros((self.pixels,self.pixels,3))
        dog_mean = np.zeros((self.pixels,self.pixels,3))

        #print(self.test_x[10])

        for i in range(len(self.test_x)):
            x_mean = np.add(x_mean,np.asarray(self.test_x[i], dtype=float) / 255) 
            if self.test_y[i] == 0:
                cat_mean = np.add(cat_mean,np.asarray(self.test_x[i], dtype=float))
            if self.test_y[i] == 1:
                dog_mean = np.add(dog_mean,np.asarray(self.test_x[i], dtype=float))
                #plt.imshow(self.test_x[i])
                #plt.title(("Dog "+ str(i)))
                #plt.show()

        fig, ax = plt.subplots(nrows=1,ncols=2)
        #print(cat_mean)
        ax[0].imshow(dog_mean/1000)
        ax[1].imshow(cat_mean/1000)
        #plt.show()

        fig, axs = plt.subplots(nrows=5, ncols=5)
        axs = axs.flat
        for i in range(25):
            idx = np.random.randint(1000,2000)
            axs[i].imshow(self.test_x[idx])  
        plt.show()
        
        

        return np.divide(x_mean,len(self.test_x))


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

    dataset.train_statistics()
