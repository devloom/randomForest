import math
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import random
import time

from tree import Tree
from dataset import Dataset


############# This is a custom built random forest #########################
############# we can compare functionality to random forest from sklearn #############
class Forest():
    def __init__(self, numTrees):
        super().__init__()
        self.numTrees = numTrees
        self.trees = [None]*self.numTrees


    def createTrees(self, dataset, full=False):
        # save the dataset
        self.ds = dataset
        # create a randomized array of the indices of the training data
        #train_length = len(dataset.train_dataset["img"])
        train_length = len(dataset.train_X)
        indices = sorted(np.array([i for i in range(train_length)]),key=lambda k:random.random())

        for i in range(self.numTrees):
            if not full:
                # give the trees a subset of the training data
                #subset = train_length//self.numTrees
                subset = train_length//5
                #indices_sub = np.array(indices[i*subset:i*subset+subset])
                indices_sub = np.array(np.random.randint(0,train_length,subset).tolist())
                tree = Tree(dataset,indices_sub)
            else:
                # tree each tree on the whole dataset
                tree = Tree(dataset,np.array(indices))
            # add all the created trees to a list
            print("Growing tree: ", i)
            self.trees[i] = tree

    def retrainTrees(self, full=False):
        # create a randomized array of the indices of the retraining data
        #train_length = len(self.ds.second_train["img"])
        train_length = len(self.ds.second_train_X)
        ## DEBUG
        print("train_length is:", train_length)

        indices = sorted(np.array([i for i in range(train_length)]),key=lambda k:random.random())

        i = 0
        for tree in self.trees:
            if not full:
                # give the trees a subset of the training data
                subset = train_length//self.numTrees
                ## DEBUG
                print("indices in forest are", indices)
                indices_sub = np.array(indices[i*subset:i*subset+subset])
                tree.retrain(indices_sub)
            else:
                # tree each tree on the whole dataset
                tree.retrain(np.array(indices))
            print("Retraining tree: ", i)
            i += 1

    def classify(self,X,y):
        num = len(X)
        # instantiate an empty list in order to accept the list of dictionaries
        class_prob_arr = [None]*self.numTrees
        # instantiate 2d array to accept the array of predicted_classes
        pred_class_arr = np.zeros((self.numTrees,num))
        # store the data from each tree
        for i in range(self.numTrees):
            pred_class_arr[i,:], class_prob_arr[i] = self.trees[i].sort(X,testing=True)

        # This list and array are meant to store the combined dictionary and prediction value per feature
        class_probs = []
        pred_class = np.zeros(num)
        for j in range(num):
            # majority vote to determine the predicted class
            counts = np.bincount(pred_class_arr[:,j].astype(int))
            pred_class[j] = np.argmax(counts)
            d_tmp = dict()
            for i in range(self.numTrees):
                d_tmp = {k: d_tmp.get(k,0) + class_prob_arr[i][j].get(k,0) for k in set(d_tmp) | set(class_prob_arr[i][j])}
            class_probs.append(d_tmp)

        return pred_class, class_probs

def main(increment=False):
    #What percentage of the available training dataset do you want to use for training? Enter [0,1]
    train_percent = 1
    #What percentage of the available testing dataset do you want to use for testing? Enter [0,1]
    test_percent = 1

    #How many classes from imagenet (up to 50)?
    num_classes = 10

    # load dataset
    dataset = Dataset(train_pct=train_percent,test_pct=test_percent,classes,fourD=False)

    ########### Create forest ############
    numTrees = 50
    forest = Forest(numTrees)

    # if we increment, split the data in the Dataset()
    if increment:
        init_classes = 5
        dataset.split_data(init_classes)

    ########### Grow trees on training data ###########
    tic = time.perf_counter()
    forest.createTrees(dataset,full=False)
    toc = time.perf_counter()
    print(f"Tree trained (grown) in {toc - tic:0.4f} seconds")

    if increment:
        ########### Retrain trees on additional data ###########
        tic = time.perf_counter()
        forest.retrainTrees(full=False)
        toc = time.perf_counter()
        print(f"Tree retrained in {toc - tic:0.4f} seconds")

    ########### Predict test data ##################
    pred_class, class_probs_dicts = forest.classify(dataset.test_X, dataset.test_y)

    length = len(pred_class)
    # Determining by majority vote
    print("Determining by majority vote")

    print(dataset.test_y[0:100])
    print(pred_class[0:100])

    num = sum([1 if dataset.test_y[i] == pred_class[i] else 0 for i in range(length)])
    print("accuracy: ", num/length)

    # Determining by class probability
    print("Determining by class probability")
    max_prob = np.zeros(length)
    for i in range(length):
        max_prob[i] = max(class_probs_dicts[i], key=class_probs_dicts[i].get)

    print(max_prob[0:100])

    num = sum([1 if dataset.test_y[i] == max_prob[i] else 0 for i in range(length)])
    print("accuracy: ", num/length)

if __name__ == '__main__':
    main()
