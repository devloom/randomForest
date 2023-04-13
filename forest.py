import math
import sklearn
import numpy as np
import matplotlib.pyplot as plt

from tree import Tree
from dataset import Dataset


############# This is a custom built random forest #########################
############# we can compare functionality to random forest from sklearn #############
class Forest():
    def __init__(self):
        super().__init__()
        self.numTrees = 1
        self.trees = []*numTrees

        self.splittingFunction = 'gini'
        #self.createTrees()


    def createTrees(self, dataset):
        for i in range(numTrees):
            tree = Tree(dataset)
            self.trees[i] = tree

    def classify(self,t):
        node = t.nodes
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class


if __name__ == '__main__':
    ############# load dataset ##########
    dataset = Dataset()
    #####################################

    ########### Create forest ############
    forest = Forest()
    
    ########### Grow trees on training data ###########
    forest.createTrees(dataset)
    ###################################################

    for i in range(forest.numTrees):
        forest.classify(self.trees[i])


    print("hallo")
    ########## Classify test data #####################
    #y_pred = forest.classify(train_img)
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
