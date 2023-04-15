from node import Node
from dataset import Dataset
from node import Node

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
import random

class Tree():
    def __init__(self,data,indices,test=False,streaming=False):
        super().__init__()

        self.max_depth = 25
        self.pixels = data.pixels
        self.increment = 5000
        
        if streaming:
            # Primarily for large datasets like ImageNet
            print("Starting the streaming algorithm")
            dataset_head = data.train_dataset.take(self.increment)
            print("We've taking the head.")
            print("Now we begin the resizing")
            resized_img = []
            for i in range(self.increment):
                resized_img.append(next(iter(dataset_head))["image"].convert("RGB").resize((data.pixels,data.pixels)))
                print("Iteration",i)

            self.train_img = resized_img
            #self.train_img = [elem["image"].convert("RGB").resize((data.pixels,data.pixels)) for elem in dataset_head] 
            print("Resizing done.")
            self.train_x = np.array([data.imgNumpy(image) for image in self.train_img])
            self.train_y = np.array(dataset_head['label'])
        else:    
            # Previous method of splitting up training data with no streamed data
            # Get randomized indicies to shuffle training data
            
            # Image resizing for training data occurs in tree
            self.train_img = [data.train_dataset[i.item()]["img"].convert("RGB").resize((data.pixels,data.pixels)) for i in indices]
            #self.train_img = [data.train_dataset[i.item()]["image"].convert("RGB").resize((data.pixels,data.pixels)) for i in indices]
            self.train_x = np.array([data.imgNumpy(image) for image in self.train_img])
            # Get correct labels using the randomized indicies
            self.train_y = np.array(data.train_dataset['label'])[indices.astype(int)]
            #self.train_y = np.array(data.train_dataset['labels'])[indices.astype(int)]
            
        

        self.classes = np.array(list(set(self.train_y)))

        self.n_classes = len(self.classes)
        

        ### test = false means we need to train the tree
        ### test = true means the tree has already been trained and we read in hyperparameters from file
        if (test == False):
            self.nodes = self.grow(self.train_x,self.train_y)
        else:
            print("Reading in tree:")
            #### code here to read in node structure of tree

    '''
    def entropy(self,p):
        #### this is informationgain function from lecture 21 slides on decision trees ########
        h = 0
        for i in range(p):
            h += 
        h = -p*np.log2(p) - (1-p)*np.log2(1-p)
        return h
    '''
        
    def grow(self,X,y,depth=0):
        
        num_samples_per_class = np.array([np.sum(y == i) for i in self.classes])
        class_probability = np.array([np.sum(y == i)/len(y) for i in self.classes])
        predicted_class = self.classes[np.argmax(num_samples_per_class)]
        #print("Tree at depth ", depth)
        #print(len(y))
        #print(class_probability)
        #print("predicted class: ", predicted_class)
        #new_classes = np.copy(self.classes)

        new_classes = np.delete(self.classes, np.where(num_samples_per_class == 0))

        #only select a subset of size sqrt(|K|) to calculate centroids
        new_classes_subset = np.random.choice(new_classes,int(np.ceil(np.sqrt(len(new_classes)))))

        #print(num_samples_per_class)
        #print(new_classes)
        node = Node(pred_class=predicted_class,class_prob=class_probability,classes=new_classes_subset,pixels=self.pixels)

        ### subarrays of X and y that correspond to new_classes subset
        X_sub = np.array([X[i] for i in range(len(X)) if (y[i] in new_classes_subset)])
        y_sub = np.array([label for label in y if (label in new_classes_subset)])



        node.splitter(X_sub, y_sub)

        #print(X)
        
        #print(node.centroids)
        #print(node.cent_split)

        
            
        indices_left = [False]*len(y)
        if node.cent_split is not None:
            # for each feature return the index of the nearest centroid where centroids are calculated for each class in the subclass array of length sqrt(|K|)
            nearest_cent_idx = np.argmin(np.array([[np.linalg.norm(X[i] - node.centroids[k]) for k in range(node.n_classes)] for i in range(len(X))]),axis=1)
            indices_left = np.array([True if np.any(np.nonzero(node.cent_split == 0)[0] == nearest_cent_idx[j]) else False for j in range(len(nearest_cent_idx))])
            X_left, y_left = X[indices_left], y[indices_left]
            X_right, y_right = X[~indices_left], y[~indices_left]
            
            
            node.left = self.grow(X_left, y_left, depth + 1)
            node.right = self.grow(X_right, y_right, depth + 1)
        return node

    def print_leaves(self,node):
        if node.left == None:  
            print ("predicted class: ", node.pred_class)
        else:
            self.print_leaves(node.left)
            self.print_leaves(node.right)
            
            

if __name__ == '__main__':
    dataset = Dataset()

    train_indices = np.array([i for i in range(0,40000,1)])
    

    tree = Tree(dataset,train_indices,streaming=False)

    node_ = tree.nodes
    
    #tree.print_leaves(node_)

    ########### accuracy on validation data #################
    pred_classes = np.zeros(len(tree.train_x))
    print("here")
    
    for i in range(len(train_indices)):
    #for i in range(1950,2050,1):
        #print(i)
        node_ = tree.nodes
        #test_img = tree.train_x[i]
        
        while node_.left:
            nearest_cent = np.argmin(np.array([np.linalg.norm(tree.train_x[i] - node_.centroids[k]) for k in range(node_.n_classes)]))
            #if (i == 2):
            #    print(nearest_cent)
            #    print(node_.cent_split[nearest_cent])
            if (node_.cent_split[nearest_cent] == 0):
                node_ = node_.left
            else:
                node_ = node_.right
        #if (i == 2):
        #    print("pred: ", node_.pred_class)
        pred_classes[i] = node_.pred_class

    print(tree.train_y[0:100])
    print(pred_classes[0:100])
    
    num = np.sum([1 if tree.train_y[i] == pred_classes[i] else 0 for i in range(len(pred_classes))])
    print("Validation accuracy: ", num/len(pred_classes))
    

    test_indices = np.array([i for i in range(10000)])
    ########### accuracy on test data #################
    pred_classes = np.zeros(len(dataset.test_x))
    print("here")
    
    for i in range(len(test_indices)):
    #for i in range(1950,2050,1):
        #print(i)
        node_ = tree.nodes
        test_img = dataset.test_x[i]
        
        while node_.left:
            nearest_cent = np.argmin(np.array([np.linalg.norm(dataset.test_x[i] - node_.centroids[k]) for k in range(node_.n_classes)]))
            #if (i == 2):
                #print(nearest_cent)
                #print(node_.cent_split[nearest_cent])
            if (node_.cent_split[nearest_cent] == 0):
                node_ = node_.left
            else:
                node_ = node_.right
        #if (i == 2):
        #    print("pred: ", node_.pred_class)
        pred_classes[i] = node_.pred_class

    print(dataset.test_y[0:100])
    print(pred_classes[0:100])
    
    num = np.sum([1 if dataset.test_y[i] == pred_classes[i] else 0 for i in range(len(pred_classes))])
    #print([1 if dataset.test_y[i] == pred_classes[i] else 0 for i in range(len(pred_classes))])
    print("Test accuracy: ", num/len(pred_classes))
    
