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
    def __init__(self, pth = "cifar10", reload=False):

        self.dataset_path = pth
        self.pixels = 32  # determines image size

        # special loading algorithim for imagenet due to size
        if pth == "imagenet-1k":
            # pick which labels we want to use
            labels = [0, 100, 200] #, 300, 400, 500, 600, 700, 800, 900]
            # STREAMING METHOD 
            train_data = load_dataset(self.dataset_path, split="train", streaming=True)
            self.train_dataset = train_data.filter(lambda img: img['label'] in labels)
            val_data = load_dataset(self.dataset_path, split="validation", streaming=True)
            self.test_dataset = val_data.filter(lambda img: img['label'] in labels)

            # NOT YET IMPLEMENTED
            #self.test_img = [image.convert("RGB").resize((self.pixels,self.pixels)) for image in self.test_dataset["img"]]
            #self.test_x = np.array([self.imgNumpy(image) for image in self.test_img])
            #self.test_y = np.array(self.test_dataset['label'])

            # SAVING THE DATA LOCALLY METHOD
            '''
            if not (os.path.isfile("../local/imagenet_train_data.hf") or reload):
                print("Whoops! Looks like you don't have the imagenet dataset downloaded yet.")
                # DIRECT LOAD METHOD (Huge datasets)
                # Instantiate empty datasets
                train_select = [] #datasets.Dataset()
                val_select = [] #datasets.Dataset()
                #load in only 5 percent at a time
                percent = 1 
                for k in np.linspace(0, 100-percent, percent*100): 
                    train_frac = load_dataset(self.dataset_path, split=f"train[{k}%:{k+percent}%]")
                    val_frac = load_dataset(self.dataset_path, split=f"validation[{k}%:{k+percent}%]")
                    # select those images which have a label in the label list
                    train_select = concatenate_datasets([train_select,train_data.filter(lambda img: img['label'] in labels)])
                    val_select = concatenate_datasets([val_select,val_data.filter(lambda img: img['label'] in labels)])

                # STREAMING METHOD (No straightforward way to save data)
                train_data = load_dataset(self.dataset_path, split="train", streaming=True)
                val_data = load_dataset(self.dataset_path, split="validation", streaming=True)
                train_select = train_data.filter(lambda img: img['label'] in labels)
                val_select = val_data.filter(lambda img: img['label'] in labels)
                # DEBUG TBAER: 4/14
                #print(next(iter(train_select)))
                #print(next(iter(val_select)))
                #Here we want to store the values in the iterator, I feel like there should be a nice solution to this
                print("Prepping for download...")
                train_select = [*train_select]
                val_select = [*val_select]

                #Saving out iterable dataset to disk after filtering for future use
                print("Saving the dataset to '../local/'")
                train_select.save_to_disk("../local/imagenet_train_data.hf")
                val_select.save_to_disk("../local/imagenet_test_data.hf")

            #Loading our saved datasets from the disk
            print("Dataset is downloaded. Loading from the disk...")
            self.train_datase = load_from_disk("../local/imagenet_train_data.hf")
            self.test_dataset = load_from_disk("../local/imagenet_test_data.hf")


            ## DEBUG TBAER: 4/14
            #single_img =  (next(iter(train_select))['image']).convert("RGB").resize((self.pixels,self.pixels))
            #print(single_img)
            #plt.imshow(single_img)
            #plt.show()

            self.test_img = [image.convert("RGB").resize((self.pixels,self.pixels)) for image in self.test_dataset["image"]]
            '''
                       
        else:
            # Load in all the data normally if not imagenet
            self.ds = load_dataset(self.dataset_path)
            self.train_dataset = self.ds['train']
            self.test_dataset = self.ds['test']
            self.test_img = [image.convert("RGB").resize((self.pixels,self.pixels)) for image in self.test_dataset["img"]]
            #self.test_img = [image.convert("RGB").resize((self.pixels,self.pixels)) for image in self.test_dataset["image"]]
            self.test_x = np.array([self.imgNumpy(image) for image in self.test_img])
            self.test_y = np.array(self.test_dataset['label'])
            #self.test_y = np.array(self.test_dataset['labels'])

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
    


if __name__ == '__main__':
    dataset = Dataset()
    #print(dataset.test_img[0])
