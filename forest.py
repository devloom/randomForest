import math
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import random

from tree import Tree
from dataset import Dataset


############# This is a custom built random forest #########################
############# we can compare functionality to random forest from sklearn #############
class Forest():
    def __init__(self):
        super().__init__()
        self.numTrees = 10
        self.trees = [None]*self.numTrees


    def createTrees(self, dataset):
        indices = sorted(np.array([i for i in range(len(dataset.train_dataset["img"]))]),key=lambda k:random.random())
        for i in range(self.numTrees):
            indices_sub = np.array(indices[i*5000:i*5000+5000])
            tree = Tree(dataset,indices_sub)
            #tree = Tree(dataset,0,30000)
            print("Growing tree: ", i)
            self.trees[i] = tree

    def classify(self,x,t):
        node_ = t.nodes      
        while node_.left:
            nearest_cent = np.argmin(np.array([np.linalg.norm(x - node_.centroids[k]) for k in range(t.n_classes)]))
            if (node_.cent_split[nearest_cent] == 0):
                node_ = node_.left
            else:
                node_ = node_.right

        return node_.pred_class, node_.class_prob
        

if __name__ == '__main__':
    ############# load dataset ##########
    dataset = Dataset()
    

    ########### Create forest ############
    forest = Forest()
    
    ########### Grow trees on training data ###########
    forest.createTrees(dataset)
    print(forest.trees)


    ########### Classify test data ##################
    pred_classes = np.zeros(len(dataset.test_x))

    for i in range(len(dataset.test_x)):
        class_vote = np.zeros(forest.trees[0].n_classes)
        class_probs = np.zeros(forest.trees[0].n_classes)
        for j in range(forest.numTrees):
            classIdx, classProb = forest.classify(dataset.test_x[i],forest.trees[j])
            if (i == 0):
                print(j, classIdx, classProb[classIdx])
            class_vote[classIdx] += 1
            class_probs = np.add(class_probs,classProb)

        #predicted class (used for top 1 accuracy)
        pred_classes[i] = np.argmax(class_probs)
        #top 5 classes by probability (used for top 5 accuracy)
        #definition here...

        if (i == 0):
            print(class_probs)
            print(pred_classes[i])

    print(dataset.test_y[0:100])
    print(pred_classes[0:100])
    num = np.sum([1 if dataset.test_y[i] == pred_classes[i] else 0 for i in range(len(pred_classes))])
    print("accuracy: ", num/len(pred_classes))
    
