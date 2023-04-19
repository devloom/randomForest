import numpy as np
import os
import matplotlib.pyplot as plt
import datasets
from datasets import load_dataset, Image
import cv2 
from scipy.cluster.vq import kmeans, vq

class Dataset:
    def __init__(self, pth = "cifar10"):

        self.dataset_path = pth
        self.pixels = 32  # determines image size

        # special loading algorithim for imagenet due to size
        if pth == "imagenet-1k":
            # pick which labels we want to use
            labels = [6, 7, 107, 340, 406, 407, 420, 471, 755, 855] 
            self.labels = labels
            self.download()

            # ONCE DATASET HAS BEEN LOADED
            #Loading our saved datasets from the disk
            print("Dataset is downloaded. Loading from the disk...")
            self.train_datase = load_from_disk("../downloads/imagenet_train_data.hf")
            self.test_dataset = load_from_disk("../downloads/imagenet_test_data.hf")

        else:
            # Load in all the data normally if not imagenet
            # self.ds = load_dataset(self.dataset_path)
            # self.train_dataset = self.ds['train']
            # self.test_dataset = self.ds['test']

                        # download the dataset
            imagenet = load_dataset(
                'frgfm/imagenette',
                'full_size',
                split='train',
                ignore_verifications=False  # set to True if seeing splits Error
            )

            images_training = []

            for n in range(0,len(imagenet)):
                # generate np arrays from the dataset images
                images_training.append(np.array(imagenet[n]['image']))

            # train_dataset = self.train_dataset
            # test_dataset = self.test_dataset

            # train_img = [np.array(train_dataset[i]['img'], dtype=float) for i in range(len(train_dataset['img']))]
            # train_img = [cv2.normalize(train_img[i], None, 0, 255, cv2.NORM_MINMAX).astype('uint8') for i in range(len(train_dataset['img']))]

            # bw_images = []
            # for img in train_img:
            #     # if RGB, transform into grayscale
            #     if len(img.shape) == 3:
            #         bw_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            #     else:
            #         # if grayscale, do not transform
            #         bw_images.append(img)

            bw_images = []
            for img in images_training:
                # if RGB, transform into grayscale
                if len(img.shape) == 3:
                    bw_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
                else:
                    # if grayscale, do not transform
                    bw_images.append(img)

            extractor = cv2.xfeatures2d.SIFT_create()

            # initialize lists where we will store *all* keypoints and descriptors
            keypoints = []
            descriptors = []

            for img in bw_images:
                # extract keypoints and descriptors for each image
                img_keypoints, img_descriptors = extractor.detectAndCompute(img, None)
                keypoints.append(img_keypoints)
                descriptors.append(img_descriptors)
            
            output_image = []
            for x in range(5):
                output_image.append(cv2.drawKeypoints(bw_images[x], keypoints[x], 0, (255, 0, 0),
                                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
                plt.imshow(output_image[x], cmap='gray')
                plt.show() 

            # set numpy seed for reproducability
            np.random.seed(0)
            # select 1000 random image index values
            sample_idx = np.random.randint(0, len(imagenet)+1, 1000).tolist()

            # extract the sample from descriptors
            # (we don't need keypoints)
            descriptors_sample = []

            for n in sample_idx:
                descriptors_sample.append(np.array(descriptors[n]))

            all_descriptors = []
            # extract image descriptor lists
            for img_descriptors in descriptors_sample:
                # extract specific descriptors within the image
                for descriptor in img_descriptors:
                    all_descriptors.append(descriptor)
            # convert to single numpy array
            all_descriptors = np.stack(all_descriptors)

            k = 200
            iters = 1
            codebook, variance = kmeans(all_descriptors, k, iters)

            visual_words = []
            for img_descriptors in descriptors:
                if img_descriptors is None:
                    continue
                else:
                # for each image, map each descriptor to the nearest codebook entry
                    img_visual_words, distance = vq(img_descriptors, codebook)
                    visual_words.append(img_visual_words)

            self.visual_words = visual_words

            frequency_vectors = []
            for img_visual_words in visual_words:
                # create a frequency vector for each image
                img_frequency_vector = np.zeros(k)
                for word in img_visual_words:
                    img_frequency_vector[word] += 1
                frequency_vectors.append(img_frequency_vector)
            # stack together in numpy array
            frequency_vectors = np.stack(frequency_vectors)

            plt.bar(list(range(k)), frequency_vectors[200])
            plt.show()


    def split_data(self,initial_num):
        # Split the labels in to the primary training set and the secondary training set
        total_labels = max(list(set(np.array(self.train_dataset['label']))))
        initial_labels = [i for i in range(initial_num)]
        second_labels = [(i+initial_num) for i in range(total_labels-initial_num+1)]
        # DEBUG 
        #print("initial_labels", initial_labels)
        #print("second_labels", second_labels)

        # split the dataset according to the chosen labels
        self.second_train = self.train_dataset.filter(lambda img: img['label'] in second_labels)
        print("Initial training has ", len(self.second_train), " number of elements")
        self.train_dataset = self.train_dataset.filter(lambda img: img['label'] in initial_labels) 
        print("Initial training has ", len(self.train_dataset), " number of elements")

        # no need to split the testing set
        #self.second_test = self.test_dataset.filter(lambda img: img['label'] in second_labels)
        #self.test_dataset = self.test_dataset.filter(lambda img: img['label'] in initial_labels)

        return

    def imgNumpy(self,img):
        img_x = np.asarray(img, dtype=float) / 255
        return img_x


    # def download(self, reload=False):
    #     # We download, preprocess, and sort the imagenet data 
    #     if not (os.path.isfile("./downloads/imagenet_test_data.hf") or reload):
    #         print("Whoops! Looks like you don't have the imagenet dataset downloaded yet.")
    #         #load in only 5 percent at a time
    #         percent = 5 
    #         self.ds = load_dataset(self.dataset_path)
    #         train_data = self.ds['train']
    #         val_data = self.ds['validation']
    #         # select those images which have a label in the label list
    #         train_select = train_data.filter(lambda img: img['label'] in self.labels)
    #         val_select = val_data.filter(lambda img: img['label'] in self.labels)
            
    #         # preprocess data
    #         train_select = train_select.map(self.transforms, batched=True)
    #         val_select = val_select.map(self.transforms, batched=True)

    #         #Saving our dataset to disk after filtering for future use
    #         print("Saving the dataset to './downloads/'")
    #         train_select.save_to_disk("./downloads/imagenet_train_data.hf")
    #         val_select.save_to_disk("./downloads/imagenet_test_data.hf")
    #     else:
    #         print("Dataset already downloaded!")
    #     return



if __name__ == '__main__':
    dataset = Dataset()
 #   dataset.download()
