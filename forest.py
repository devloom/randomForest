import math
import sklearn
import numpy as np
import matplotlib.pyplot as plt

from tree import Tree
from dataset import Dataset


############# This is a custom built random forest #########################
############# we can compare functionality to random forest built from sklearn #############
class Forest():
    def __init__(self):
        super().__init__()
        self.numTrees = 1
        self.trees = []*numTrees
        self.splittingFunction = 'gini'
        #self.createTrees()


    def createTrees(self, dataset):
        for i in range(numTrees):
            tree = Tree()
            tree.setSplittingFunction(self.splittingFunction)
            tree.grow()
            self.trees[i] = tree

    def classify():
        for i in self.trees:
            dec_path = [False]*i.depth
            for j in range(i.depth):
                dec_path[j] = i.node(i.nodes[j])




if __name__ == '__main__':
    ############# load dataset ##########
    dataset = Dataset()


    '''
    X_train, y_train = loadData()
    length = 120
    x_mean = np.zeros((length,3))
    x_std = np.zeros((length,3))
    y_label = np.zeros(length)
    for i in range(length):
        train_img = dataNumpy(X_train[i])
        y_label[i] = y_train[i] 
        y_pred = forest.classify(train_img)

        x_mean[i] = np.mean(train_img, axis=(0))
        #x_mean = np.mean(X_train, axis=(0,1))  # per-channel mean
        x_std[i] = np.std(train_img, axis=(0))
    '''
    #####################################

    ########### Create forest ############
    forest = Forest()
    
    ########### Grow trees on training data ###########
    forest.createTrees(dataset)
    ###################################################

    ########## Classify test data #####################
    y_pred = forest.classify(train_img)
    ###################################################
        
    
    
    

    '''
    print(x_mean)
    print(y_label)
    
    classA = np.array([[True if y_train[i] == 57 else False for i in range(length)]])
    classB = np.array([[True if y_train[i] == 5 else False for i in range(length)]])
    classC = np.array([[True if y_train[i] == 127 else False for i in range(length)]])
    #classA = y_train == 57
    #classB = y_train == 5

    #plt.scatter(x_mean[0, classA[0, :]], x_mean[1, classA[0, :]], color='b', s=0.3)
    #plt.scatter(x_mean[0, classB[0, :]], x_mean[1, classB[0, :]], color='r', s=0.3)
    plt.scatter(x_std[classA[0, :], 0], x_std[classA[0, :], 1], color='b', s=1.5)
    plt.scatter(x_std[classB[0, :], 0], x_std[classB[0, :], 1], color='r', s=1.5)
    plt.scatter(x_std[classC[0, :], 0], x_std[classC[0, :], 1], color='g', s=1.5)
    #print (X_train[0])
    print (x_mean)
    print (x_std)
    plt.show()
    '''