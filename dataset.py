import numpy as np
import os
import matplotlib.pyplot as plt
import datasets
from datasets import load_dataset, Image, concatenate_datasets

# PREVIOUS DATASETS
#"keremberke/pokemon-classification",
#"Bingsu/Cat_and_Dog"
#"jbarat/plant_species"

# CURRENT DATASET
#"cifar10"

class Dataset:
    def __init__(self, pth = "cifar10"):

        self.dataset_path = pth
        self.pixels = 32  # determines image size

        # special loading algorithim for imagenet due to size
        if pth == "imagenet-1k":
            # pick which labels we want to use
            self.labels = [6, 7, 107, 340, 406, 407, 420, 471, 755, 855] 
            # STREAMING METHOD 
            train_data = load_dataset(self.dataset_path, split="train", streaming=True)
            self.train_dataset = train_data.filter(lambda img: img['label'] in labels)
            val_data = load_dataset(self.dataset_path, split="validation", streaming=True)
            self.test_dataset = val_data.filter(lambda img: img['label'] in labels)

            # NOT YET IMPLEMENTED
            #self.test_img = [image.convert("RGB").resize((self.pixels,self.pixels)) for image in self.test_dataset["img"]]
            #self.test_x = np.array([self.imgNumpy(image) for image in self.test_img])
            #self.test_y = np.array(self.test_dataset['label'])

            ''' ONCE DATASET HAS BEEN LOADED
            #Loading our saved datasets from the disk
            print("Dataset is downloaded. Loading from the disk...")
            self.train_datase = load_from_disk("../local/imagenet_train_data.hf")
            self.test_dataset = load_from_disk("../local/imagenet_test_data.hf")
            '''

                       
        else:
            # Load in all the data normally if not imagenet
            self.ds = load_dataset(self.dataset_path)
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
        ## DEBUG LINES from dog/cat data
        #cat_mean = np.zeros((self.pixels,self.pixels,3))
        #dog_mean = np.zeros((self.pixels,self.pixels,3))
        for i in range(len(self.test_x)):
            x_mean = np.add(x_mean,np.asarray(self.test_x[i], dtype=float) / 255) 
            ## DEBUG LINES from dog/cat data
            #if self.test_y[i] == 0:
            #    cat_mean = np.add(cat_mean,np.asarray(self.test_x[i], dtype=float))
            #if self.test_y[i] == 1:
            #    dog_mean = np.add(dog_mean,np.asarray(self.test_x[i], dtype=float))
        #fig, ax = plt.subplots(nrows=1,ncols=2)
        #ax[0].imshow(dog_mean/1000)
        #ax[1].imshow(cat_mean/1000)
        #fig, axs = plt.subplots(nrows=5, ncols=5)
        #axs = axs.flat
        #for i in range(25):
        #    idx = np.random.randint(1000,2000)
        #    axs[i].imshow(self.test_x[idx])  
        #plt.show()

        return np.divide(x_mean,len(self.test_x))

    def download(self, reload=False):
        # We download, preprocess, and sort the imagenet data 
        if not (os.path.isfile("./data/imagenet_train_data.hf") or reload):
            print("Whoops! Looks like you don't have the imagenet dataset downloaded yet.")
            #load in only 5 percent at a time
            percent = 5 
            self.ds = load_dataset(self.dataset_path)
            train_data = self.ds['train']
            val_data = self.ds['test']
            # select those images which have a label in the label list
            train_select = train_data.filter(lambda img: img['label'] in labels)
            val_select = val_data.filter(lambda img: img['label'] in labels)

            #Saving our dataset to disk after filtering for future use
            print("Saving the dataset to './data/'")
            train_select.save_to_disk("./data/imagenet_train_data.hf")
            val_select.save_to_disk("./data/imagenet_test_data.hf")
        return
    


if __name__ == '__main__':
    dataset = Dataset()
    dataset.download()
