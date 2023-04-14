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
        self.numTrees = 10
        self.trees = [None]*self.numTrees
        print(self.trees)

        self.splittingFunction = 'gini'
        #self.createTrees()


    def createTrees(self, dataset):
        for i in range(self.numTrees):
            tree = Tree(dataset,i*5000,i*5000+5000)
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
        #node = t.nodes
        #while node.left:
        #    if inputs[node.feature_index] < node.threshold:
        #        node = node.left
        #    else:
        #        node = node.right
        




if __name__ == '__main__':
    ############# load dataset ##########
    dataset = Dataset()
    #####################################

    ########### Create forest ############
    forest = Forest()
    
    ########### Grow trees on training data ###########
    forest.createTrees(dataset)
    print(forest.trees)
    ###################################################

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
        
        pred_classes[i] = np.argmax(class_probs)
        if (i == 0):
            print(class_probs)
            print(pred_classes[i])
        #if (i == 0):
        #    print(class_vote)
        #    print(pred_classes[i])

    print(dataset.test_y[0:100])
    print(pred_classes[0:100])
    num = np.sum([1 if dataset.test_y[i] == pred_classes[i] else 0 for i in range(len(pred_classes))])
    print("accuracy: ", num/len(pred_classes))
    ##################################################
    
    
    
    

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