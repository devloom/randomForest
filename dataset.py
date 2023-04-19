import numpy as np
import os
import matplotlib.pyplot as plt
import datasets
from datasets import load_dataset, Image
import cv2
from scipy.cluster.vq import kmeans,vq
# PREVIOUS DATASETS
#"keremberke/pokemon-classification",
#"Bingsu/Cat_and_Dog"
#"jbarat/plant_species"

# CURRENT DATASET
#"cifar10"

class Dataset:
    def __init__(self, train_pct, test_pct, pth = "cifar10"):

        self.dataset_path = pth
        self.pixels = 32  # determines image size
        self.bags = None

        # special loading algorithim for imagenet due to size
        if pth == "imagenet-1k":
            # pick which labels we want to use
            labels = [6, 7, 107, 340, 406, 407, 420, 471, 755, 855] 
            self.labels = labels
            self.download()

            # ONCE DATASET HAS BEEN LOADED
            #Loading our saved datasets from the disk
            print("Dataset is downloaded. Loading from the disk...")
            self.train_dataset = load_from_disk("../downloads/imagenet_train_data.hf")
            self.test_dataset = load_from_disk("../downloads/imagenet_test_data.hf")

        else:
            # Load in all the data normally if not imagenet
            self.ds = load_dataset(self.dataset_path)

            ##### Take randomized subset of train and test dataset as specified by user ##########
            train_ds_length = self.ds['train'].num_rows
            test_ds_length = self.ds['test'].num_rows
            train_length = int(train_ds_length*train_pct)
            test_length = int(test_ds_length*test_pct)
            train_idx = np.random.randint(0, train_ds_length, train_length).tolist()
            test_idx = np.random.randint(0, test_ds_length, test_length).tolist()

            
            self.train_dataset = self.ds['train'][train_idx]
            self.test_dataset = self.ds['test'][test_idx]

            #print(self.train_dataset)

            ########## DEPRECATED - BEFORE VISUAL BAG OF WORDS ################
            
            self.train_img = [self.train_dataset["img"][i].convert("RGB").resize((self.pixels,self.pixels)) for i in range(train_length)]
            self.train_X = np.array([self.imgNumpy(image) for image in self.train_img])
            self.train_y = np.array([self.train_dataset['label'][i] for i in range(train_length)])
            #self.train_y = np.array(self.train_dataset['label'])[indices.astype(int)]

            #self.test_img = [image.convert("RGB").resize((self.pixels,self.pixels)) for image in self.test_dataset["img"]]
            self.test_img = [self.test_dataset["img"][i].convert("RGB").resize((self.pixels,self.pixels)) for i in range(test_length)]
            self.test_X = np.array([self.imgNumpy(image) for image in self.test_img])
            self.test_y = np.array([self.test_dataset['label'][i] for i in range(test_length)])
            #self.test_y = np.array(self.test_dataset['label'])
            
            print(self.train_X.shape)
            print(self.train_y.shape)
            print(self.test_X.shape)
            '''
            ########## USING BAG OF WORDS #######################
            print("Loading dataset and converting images to bag of words features. This may take a while...")
            #call function to return images as bag of visual words
            self.train_X, self.train_y = self.sift(self.train_dataset)
            self.test_X, self.test_y = self.sift(self.test_dataset)
            print("Finished loading dataset.")
            '''

    def split_data(self,initial_num):
        # Split the labels in to the primary training set and the secondary training set
       


        #print(set(np.array(self.train_dataset['label'])))
        total_labels = max(list(set(np.array(self.train_dataset['label']))))
        initial_labels = [i for i in range(initial_num)]
        second_labels = [(i+initial_num) for i in range(total_labels-initial_num+1)]
        # DEBUG 
        #print("initial_labels", initial_labels)
        #print("second_labels", second_labels)

        initial_idx = np.where(np.logical_and(self.train_y>=initial_labels[0], self.train_y<=initial_labels[-1]))[0].tolist()
        second_idx = np.where(np.logical_and(self.train_y>=second_labels[0], self.train_y<=second_labels[-1]))[0].tolist()


        self.second_train_X = self.train_X[second_idx]
        self.second_train_y = self.train_y[second_idx]
        self.train_X = self.train_X[initial_idx]
        self.train_y = self.train_y[initial_idx]
        

        # split the dataset according to the chosen labels
        '''
        self.second_train = self.train_dataset.filter(lambda img: img['label'] in second_labels)
        print("Initial training has", len(self.second_train), "number of elements")
        self.train_dataset = self.train_dataset.filter(lambda img: img['label'] in initial_labels) 
        print("Retraining training has", len(self.train_dataset), "number of elements")
        '''

        # no need to split the testing set
        #self.second_test = self.test_dataset.filter(lambda img: img['label'] in second_labels)
        #self.test_dataset = self.test_dataset.filter(lambda img: img['label'] in initial_labels)

        return

    def imgNumpy(self,img):
        img_x = np.asarray(img, dtype=float) / 255
        return img_x


    def sift(self,data):
        img = [np.array(data['img'][i], dtype=float) for i in range(len(data['img']))]
        ##################### 2) normalize numpy arrays and make 8 bit integer data type #####################
        img = [cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8') for image in img]

        y = [np.array(data['label'][i], dtype=int) for i in range(len(data['img']))]

        '''
        bw_images = []
        for img in train_img:
            # if RGB, transform into grayscale
            if len(img.shape) == 3:
                bw_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            else:
                # if grayscale, do not transform
                bw_images.append(img)
        '''
        bw_images = img

        ################### 3) Use SIFT to extract features (keypoints, descriptors) ########################
        # defining feature extractor that we want to use (SIFT)
        extractor = cv2.xfeatures2d.SIFT_create()

        # initialize lists where we will store *all* keypoints and descriptors
        keypoints = []
        descriptors = []
        i = 0
        for img in bw_images:
            # extract keypoints and descriptors for each image
            img_keypoints, img_descriptors = extractor.detectAndCompute(img, None)
            if img_descriptors is None:
                # del i element of y
                y.pop(i)
                continue
            keypoints.append(img_keypoints)
            descriptors.append(img_descriptors)
            i += 1


        ################## 4) Stack descriptors together for all training features (or subset depending on time concerns) ##################
        np.random.seed(0)
        # select 1000 random image index values
        sample_idx = np.random.randint(0, len(descriptors), len(descriptors)).tolist()

        # extract the sample from descriptors
        # (we don't need keypoints)
        descriptors_sample = []

        for n in sample_idx:
            descriptors_sample.append(np.array(descriptors[n]))

        #print(descriptors_sample)
        all_descriptors = []
        # extract image descriptor lists
        i = 0
        for img_descriptors in descriptors_sample:
            #print(i)
            #print(img_descriptors)
            i += 1
            # extract specific descriptors within the image
            for descriptor in img_descriptors:
                all_descriptors.append(descriptor)
        # convert to single numpy array
        all_descriptors = np.stack(all_descriptors)


        ######################### 5) Use k means to make codebook which converts image features to word features ######################
        # perform k-means clustering to build the codebook

        self.bags = k = 100
        iters = 1
        codebook, variance = kmeans(all_descriptors, k, iters)


        ######################## 6) Convert all training image features to bag of visual words ######################

        visual_words = []
        for img_descriptors in descriptors:
            # for each image, map each descriptor to the nearest codebook entry
            img_visual_words, distance = vq(img_descriptors, codebook)
            visual_words.append(img_visual_words)


        ####################### 7) Make sparse vectors out of visual words (so we have x number of features (x is
        #######################    length of training dataset) each of length k)
        frequency_vectors = []
        for img_visual_words in visual_words:
            # create a frequency vector for each image
            img_frequency_vector = np.zeros(k)
            for word in img_visual_words:
                img_frequency_vector[word] += 1
            frequency_vectors.append(img_frequency_vector)
        # stack together in numpy array
        frequency_vectors = np.stack(frequency_vectors)

        X = np.array(frequency_vectors)
        y = np.array(y)

        return X, y

    def train_statistics(self):
        x_mean = np.zeros((self.pixels,self.pixels,3))
        ## DEBUG LINES from dog/cat data
        #cat_mean = np.zeros((self.pixels,self.pixels,3))
        #dog_mean = np.zeros((self.pixels,self.pixels,3))
        for i in range(len(self.test_X)):
            x_mean = np.add(x_mean,np.asarray(self.test_X[i], dtype=float) / 255) 
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

        return np.divide(x_mean,len(self.test_X))

    def download(self, reload=False):
        # We download, preprocess, and sort the imagenet data 
        if not (os.path.exists("./downloads/imagenet_test_data.hf") or reload):
            print("Whoops! Looks like you don't have the imagenet dataset downloaded yet.")
            #load in only 5 percent at a time
            percent = 5 
            self.ds = load_dataset(self.dataset_path)
            train_data = self.ds['train']
            val_data = self.ds['validation']
            # select those images which have a label in the label list
            train_select = train_data.filter(lambda img: img['label'] in self.labels)
            val_select = val_data.filter(lambda img: img['label'] in self.labels)
            
            # preprocess data
            train_select = train_select.map(self.transforms, batched=True)
            val_select = val_select.map(self.transforms, batched=True)

            #Saving our dataset to disk after filtering for future use
            print("Saving the dataset to './downloads/'")
            train_select.save_to_disk("./downloads/imagenet_train_data.hf")
            val_select.save_to_disk("./downloads/imagenet_test_data.hf")
        else:
            print("Dataset already downloaded!")
        return
            

    def transforms(self,data):
        data["image"] = [image.convert("RGB").resize((self.pixels,self.pixels)) for image in data["image"]]
        return data



if __name__ == '__main__':
    #What percentage of the available training dataset do you want to use for training? Enter [0,1]
    train_percent = 0.1
    #What percentage of the available testing dataset do you want to use for testing? Enter [0,1]
    test_percent = 0.1

    dataset = Dataset(train_pct=train_percent,test_pct=test_percent)
    init_classes = 5
    dataset.split_data(init_classes)
    #dataset.download()
