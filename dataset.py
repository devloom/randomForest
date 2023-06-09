import os
import cv2
import time
import pickle
import datasets
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset, Image, load_from_disk
from scipy.cluster.vq import kmeans,vq
# PREVIOUS DATASETS
#"keremberke/pokemon-classification",
#"Bingsu/Cat_and_Dog"
#"jbarat/plant_species"

# CURRENT DATASET
#"cifar10"

class Dataset:
    def __init__(self, train_pct=1.0, test_pct=1.0, numclasses = 50, pth = "imagenet-1k", recalc=False, fourD=False):

        self.dataset_path = pth
        self.pixels = 32  # determines image size
        self.image_name = 'image'
        self.label_name = 'label'
        self.test_name = 'validation'
        self.fourD = fourD
        self.recalc = recalc
        if pth == "imagenet-1k":
            self.pixels = 224
            self.image_name = 'image'
            self.label_name = 'label'
            self.test_name = 'validation'
        elif pth == "cifar10":
            self.pixels = 64
            self.image_name = 'img'
            self.label_name = 'label'
            self.test_name = 'test'
        elif pth == "cifar100":
            self.pixels = 64
            self.image_name = 'img'
            self.label_name = 'fine_label'
            self.test_name = 'test'
        elif pth == "frgfm/imagenette":
            self.pixels = 320
            self.image_name = 'image'
            self.label_name = 'label'
            self.test_name = 'test'
        self.bags = None

        if pth == "imagenet-1k":
            # due to imagenet's large size. We filter by label and then save to file
            # pick which labels we want to use
            labels = [2,5,9,15,22,25,72,98,134,208,275,279,331,335,350,364,366,375,400,406,425,435,451,454,477,483,502,507,533,563,566,572,574,587,600,606,643,646,663,671,708,717,727,739,750,773,810,836,942,986]
            print("You have selected", len(labels), "labels")
            self.labels = labels
            self.download()
        elif pth == "frgfm/imagenette":
            self.ds = load_dataset(self.dataset_path,'320px')
        else:
        # Load in all the data normally if not imagenet
            self.ds = load_dataset(self.dataset_path)

        if fourD:
            print("Loading datasets as converting images to 3D arrays. This may take a while...")
            ##### Take randomized subset of train and test dataset as specified by user ##########
            train_ds_length = self.ds['train'].num_rows
            test_ds_length = self.ds[self.test_name].num_rows
            train_length = int(train_ds_length*train_pct)
            test_length = int(test_ds_length*test_pct)
            train_idx = np.random.randint(0, train_ds_length, train_length).tolist()
            test_idx = np.random.randint(0, test_ds_length, test_length).tolist()

            self.train_dataset = self.ds['train'][train_idx]
            self.test_dataset = self.ds[self.test_name][test_idx]
            ########## DEPRECATED - BEFORE VISUAL BAG OF WORDS ################
            tic = time.perf_counter()
            self.train_img = [self.train_dataset[self.image_name][i].convert("RGB").resize((self.pixels,self.pixels)) for i in range(train_length)]
            self.train_X = np.array([self.imgNumpy(image) for image in self.train_img])
            self.train_y = np.array([self.train_dataset[self.label_name][i] for i in range(train_length)])
            toc = time.perf_counter()
            print(f"Training dataset loaded and codebook generated in {toc - tic:0.4f} seconds")

            tic = time.perf_counter()
            self.test_img = [self.test_dataset[self.image_name][i].convert("RGB").resize((self.pixels,self.pixels)) for i in range(test_length)]
            self.test_X = np.array([self.imgNumpy(image) for image in self.test_img])
            self.test_y = np.array([self.test_dataset[self.label_name][i] for i in range(test_length)])
            toc = time.perf_counter()
            print(f"Testing dataset loaded in {toc - tic:0.4f} seconds")
        else:
            ########## USING BAG OF WORDS #######################
            if not (os.path.isfile("./wordBags/"+pth+".npz")) or recalc:
                print("Loading datasets and converting images to bag of words features. This may take a while...")
                ##### Take randomized subset of train and test dataset as specified by user ##########
                train_ds_length = self.ds['train'].num_rows
                test_ds_length = self.ds[self.test_name].num_rows
                train_length = int(train_ds_length*train_pct)
                test_length = int(test_ds_length*test_pct)
                train_idx = np.random.randint(0, train_ds_length, train_length).tolist()
                test_idx = np.random.randint(0, test_ds_length, test_length).tolist()

                self.train_dataset = self.ds['train'][train_idx]
                self.test_dataset = self.ds[self.test_name][test_idx]

                #call function to return images as bag of visual words
                print("Loading training dataset")
                tic = time.perf_counter()
                self.train_X, self.train_y = self.sift(self.train_dataset,Train=True)
                toc = time.perf_counter()
                print(f"Training dataset loaded and codebook generated in {toc - tic:0.4f} seconds")
                print("Loading testing dataset")
                tic = time.perf_counter()
                self.test_X, self.test_y = self.sift(self.test_dataset)
                toc = time.perf_counter()
                print(f"Testing dataset loaded in {toc - tic:0.4f} seconds")
                print("Finished loading datasets.")
                # create a file to save the converted features in
                outfile = "./wordBags/"+pth+".npz"
                print("Saving bag of words")
                # Save the codebook to a npzfile
                np.savez(outfile, train_X=self.train_X,train_y=self.train_y,test_X=self.test_X,test_y=self.test_y,bags=self.bags)
            else:
                # load the features from file
                print("Loading bag of words")
                npzfile = np.load("./wordBags/"+pth+".npz")
                self.train_X, self.train_y, self.test_X, self.test_y, self.bags = npzfile['train_X'], npzfile['train_y'], npzfile['test_X'], npzfile['test_y'], npzfile['bags']


        ### only take num_classes from train and test data
        tot_classes = np.array(list(set(self.train_y)))
        if (numclasses > 50 or numclasses < 2):
            print("Please select number of classes between 2 and 50. Setting to 50.")
        elif(numclasses < 50):
            #class_labels = np.array(np.random.randint(0,tot_classes,numclasses).tolist())
            class_labels = np.random.choice(tot_classes,numclasses,replace=False)
            print(tot_classes)
            print(class_labels)
            train_idx = np.array([i for i in range(len(self.train_y)) if self.train_y[i] in class_labels])
            self.train_X = np.array(self.train_X[train_idx])
            self.train_y = np.array(self.train_y[train_idx])
            test_idx = np.array([i for i in range(len(self.test_y)) if self.test_y[i] in class_labels])
            self.test_X = np.array(self.test_X[test_idx])
            self.test_y = np.array(self.test_y[test_idx])
            #print(train_idx)





    def split_data(self,initial_num):
        # Split the labels in to the primary training set and the secondary training set
        #print(set(np.array(self.train_dataset['label'])))
        #total_labels = max(list(set(np.array(self.train_dataset['label']))))
        total_labels = list(set(self.train_y))
        total_labels.sort()
        #### WARNING: OVERWRITING INITIAL_NUM IN UNEXPECTED WAY
        #initial_num = len(total_labels)//2
        # DEBUG
        print("total_labels", total_labels)
        #coarse_label
        initial_labels = [total_labels[i] for i in range(initial_num)]
        second_labels = [total_labels[i+initial_num] for i in range(len(total_labels)-initial_num)]
        # DEBUG
        print("initial_labels", initial_labels)
        print("second_labels", second_labels)

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


    def generateCodebook(self, descriptors):
        ################## 4) Stack descriptors together for all training features (or subset depending on time concerns) ##################
        np.random.seed(0)
        # select 1000 random image index values
        num_descriptors = min(2000, len(descriptors))
        sample_idx = np.random.randint(0, len(descriptors), num_descriptors).tolist()

        # extract the sample from descriptors
        # (we don't need keypoints)
        descriptors_sample = []

        for n in sample_idx:
            descriptors_sample.append(np.array(descriptors[n]))

        #print(descriptors_sample)
        all_des = []
        # extract image descriptor lists
        i = 0
        for img_descriptors in descriptors_sample:
            #print(i)
            #print(img_descriptors)
            i += 1
            # extract specific descriptors within the image
            for descriptor in img_descriptors:
                all_des.append(descriptor)
        # convert to single numpy array
        all_des = np.stack(all_des)


        ######################### 5) Use k means to make codebook which converts image features to word features ######################
        # perform k-means clustering to build the codebook


        self.bags = k = 1000
        iters = 1
        cb, var = kmeans(all_des, k, iters)



        return cb, var

    def sift(self,data,Train=False):
        #img = [np.array(data['img'][i], dtype=float) for i in range(len(data['img']))]
        img = [np.array(data[self.image_name][i].convert("RGB").resize((self.pixels,self.pixels)), dtype=float) for i in range(len(data[self.image_name]))]
        ##################### 2) normalize numpy arrays and make 8 bit integer data type #####################
        images = [cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8') for image in img]

        y = [np.array(data[self.label_name][i], dtype=int) for i in range(len(data[self.image_name]))]



        ################### 3) Use SIFT to extract features (keypoints, descriptors) ########################
        # defining feature extractor that we want to use (SIFT)
        extractor = cv2.xfeatures2d.SIFT_create()

        # initialize lists where we will store *all* keypoints and descriptors
        keypoints = []
        descriptors = []
        i = 0
        print("Extracting features")
        for img in images:
            if i%1000 == 0:
                print("Image", i)
            # extract keypoints and descriptors for each image
            img_keypoints, img_descriptors = extractor.detectAndCompute(img, None)
            if img_descriptors is None:
                # del i element of y
                y.pop(i)
                continue
            keypoints.append(img_keypoints)
            descriptors.append(img_descriptors)
            i += 1

        ################## If training dataset, generate codebook using kmeans ####################################
        if Train:
            print("Generating codebook")
            self.codebook, self.variance = self.generateCodebook(descriptors)


        ######################## 6) Convert all training image features to bag of visual words ######################

        visual_words = []
        for img_descriptors in descriptors:
            # for each image, map each descriptor to the nearest codebook entry
            img_visual_words, distance = vq(img_descriptors, self.codebook)
            visual_words.append(img_visual_words)


        ####################### 7) Make sparse vectors out of visual words (so we have x number of features (x is
        #######################    length of training dataset) each of length k)
        freq = []
        for img_visual_words in visual_words:
            # create a frequency vector for each image
            img_freq = np.zeros(self.bags)
            for word in img_visual_words:
                img_freq[word] += 1
            freq.append(img_freq)
        # stack together in numpy array
        freq = np.stack(freq)

        df = np.sum(freq > 0, axis=0)
        idf = np.log(len(freq)/ df)
        tfidf = freq * idf

        X = np.array(tfidf)
        y = np.array(y)

        return X, y

    def train_statistics(self):
        x_mean = np.zeros((self.pixels,self.pixels,3))
        for i in range(len(self.test_X)):
            x_mean = np.add(x_mean,np.asarray(self.test_X[i], dtype=float) / 255)

        return np.divide(x_mean,len(self.test_X))

    def download(self, reload=False):
        '''
        # We download, preprocess, and sort the imagenet data
        if not (os.path.exists("./downloads/imagenet_data.hf") or reload):
            print("Whoops! Looks like you don't have the imagenet dataset downloaded yet.")
            self.ds = load_dataset(self.dataset_path)
            # select those images which have a label in the label list
            dataset_select = self.ds.filter(lambda img: img['label'] in self.labels)
            # preprocess data
            self.ds = dataset_select.map(self.transforms, batched=True)

            #Saving our dataset to disk after filtering for future use
            print("Saving the dataset to './downloads/'")
            self.ds.save_to_disk("./downloads/imagenet_data.hf")

            #Overwrite the dataset_dict.json to exclude test data.
            new_dataset_dict = {"splits": ["train", "validation"]}
            json_object = json.dumps(new_dataset_dict, indent=4)
            with open("./downloads/imagenet_data.hf/dataset_dict.json", "w") as outfile:
                outfile.write(json_object)
        else:
        '''
            # ONCE DATASET HAS BEEN LOADED
            #Load our saved datasets from the disk
        if not (os.path.isfile("./wordBags/"+self.dataset_path+".npz")) or self.recalc:
            print("Dataset is downloaded. Loading from the disk...")
            self.ds = load_from_disk("./downloads/imagenet_data.hf")
        else:
            print("You already have the bag of words calculated, run dataset.py if you want to recalculate it")

        return


    def transforms(self,data):
        data["image"] = [image.convert("RGB").resize((self.pixels,self.pixels)) for image in data["image"]]
        return data



if __name__ == '__main__':
    #What percentage of the available training dataset do you want to use for training? Enter [0,1]
    train_percent = 1.0
    #What percentage of the available testing dataset do you want to use for testing? Enter [0,1]
    test_percent = 1.0
    # recalc asks if you want to recalculate the codebook
    dataset = Dataset(train_pct=train_percent,test_pct=test_percent, recalc=True)
    # splitting the dataset
    init_classes = 5
    dataset.split_data(init_classes)
    # if you want to downlaod the iamgenet-1k data
    #dataset.download()
