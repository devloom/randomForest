from node import Node
from dataset import Dataset
from node import Node
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from tqdm import tqdm 
import random


@njit
def index(array, item):
    for idx, val in np.ndenumerate(array):
        if val == item:
            return idx

class Tree():
    def __init__(self,data,indices,test=False,streaming=False):
        super().__init__()
        # Setup information
        self.max_depth = 25
        self.pixels = data.pixels
        self.indices = indices
        self.increment = 5000
        # for use in resizing
        self.data = data
        
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
            # Image resizing for training data occurs in tree
            self.train_img = [data.train_dataset[i.item()]["img"].convert("RGB").resize((data.pixels,data.pixels)) for i in indices]
            #self.train_img = [data.train_dataset[i.item()]["image"].convert("RGB").resize((data.pixels,data.pixels)) for i in indices]
            self.train_x = np.array([data.imgNumpy(image) for image in self.train_img])
            self.train_y = np.array(data.train_dataset['label'])[indices.astype(int)]

        
        # for use in grow
        self.classes = np.array(list(set(self.train_y)))
        self.n_classes = len(self.classes)
        # DEBUG 
        #print("self.classes:", self.classes)

        ### test = false means we need to train the tree
        ### test = true means the tree has already been trained and we read in hyperparameters from file
        if (test == False):
            print("Growing...")
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


        # number of samples
        # WARNING!!! Encountering len(y) = 0, causes a runtime error
        num = len(y)
        # find the class probabilites 
        num_samples_per_class = np.array([np.sum(y == i) for i in self.classes])
        class_probability = num_samples_per_class/num
        predicted_class = self.classes[np.argmax(num_samples_per_class)]

        # DEBUG
        #print("for a node of depth", depth)
        #print("num of samples per class", num_samples_per_class)
        #print("class prob:", class_probability)
        #print("we predict class", predicted_class)

        # Delete the classes which have no associated samples and take a subset
        new_classes = np.delete(self.classes, np.where(num_samples_per_class == 0))
        new_classes_subset = np.random.choice(new_classes,int(np.ceil(np.sqrt(len(new_classes)))),replace=False)

        X_sub = np.array([X[i] for i in range(len(X)) if (y[i] in new_classes_subset)])
        y_sub = np.array([label for label in y if (label in new_classes_subset)])

        # Create a node, set attributes and find the splitting function
        node = Node()
        node.pred_class = predicted_class
        node.class_prob = class_probability
        node.classes = new_classes_subset
        node.n_classes = len(new_classes_subset)
        node.pixels = self.pixels
        node.depth = depth
        node.splitter(X_sub, y_sub)

        # From the splitting function & centroids assing the data to go to either the left or right right node
        indices_left = [False]*num
        if node.cent_split is not None:
            # for each feature return the index of the nearest centroid where centroids are calculated for each class in the subclass array of length sqrt(|K|)
            nearest_cent_idx = np.argmin(np.array([[np.linalg.norm(X[i] - node.centroids[k]) for k in range(node.n_classes)] for i in range(num)]),axis=1)
            indices_left = np.array([True if np.any(np.nonzero(node.cent_split == 0)[0] == nearest_cent_idx[j]) else False for j in range(len(nearest_cent_idx))])
            X_left, y_left = X[indices_left], y[indices_left]
            X_right, y_right = X[~indices_left], y[~indices_left]
           
            # Recursively call grow on the left and right daughters
            node.left = self.grow(X_left, y_left, depth + 1)
            (node.left).parent = node
            (node.left).branch = "left"
            node.right = self.grow(X_right, y_right, depth + 1)
            (node.right).parent = node
            (node.right).branch = "right"
        return node

    def print_leaves(self,node):
        if node.left == None:  
            print ("predicted class: ", node.pred_class)
        else:
            self.print_leaves(node.left)
            self.print_leaves(node.right)

    def retrain(self):
        # import the data
        data = self.data
        # We combine the new data with the old training data for retraining
        new_train_img = [data.second_train[i.item()]["img"].convert("RGB").resize((self.pixels,self.pixels)) for i in self.indices]
        new_x = np.array([data.imgNumpy(image) for image in new_train_img])
        new_y = np.array(data.second_train['label'])[self.indices.astype(int)]
        # find daughters of the root node
        node_list = (self.nodes).find_daughters()
        # DEBUG 
        print("node list", len(node_list))

        # uniform probability of retraining each node (currently at 0%)
        retrain_nodes = [node for node in node_list if (np.random.uniform() >= 0.95)]
        print("retraining selection including daughters", len(retrain_nodes))
        # remove the daughter nodes to avoid double retaining nodes
        for node in retrain_nodes:
            node_daughters = node.find_daughters()
            retrain_nodes = [node for node in retrain_nodes if node not in node_daughters]
            # Delete the nodes which will be trained over in retraining
            for condemned in node_daughters:
                del condemned
        # Delete the training data from the nodes that won't be retrained
        for node in node_list: 
            if node not in retrain_nodes:
                node.X = None
                node.y = None
        # DEBUG    
        print("retraining selection final", len(retrain_nodes))
        i = 0
        for node in retrain_nodes:
            # DEBUG
            print("Retrained node:", i)
            i += 1
            comb_train_x = np.concatenate((node.X,new_x))
            comb_train_y = np.concatenate((node.y,new_y))                
            # Reset self.classes for the larger dataset
            #self.classes = np.array(list(set(comb_train_y)))
            #self.n_classes = len(self.classes)
            # We reassign the retrained node to the position it occupied in its parent
            if node.branch == "left":
                (node.parent).left = self.grow(comb_train_x, comb_train_y, depth=node.depth)
            elif node.branch == "right":
                (node.parent).right = self.grow(comb_train_x, comb_train_y, depth=node.depth)
            else:
                print("node.branch was undefined")
                break


def main(increment=True):
    # Initialize dataset
    dataset = Dataset()

    # arbitrary indicies determine how large the training dataset is
    train_indices = np.array([i for i in range(0,20000,1)])


    if increment:
        # split the data for incrmental learning
        init_classes = 5
        dataset.split_data(init_classes)
        # Train tree on initial data and calculate nodes (node_ is the root node)
        print("Training tree on original data")
        tree = Tree(dataset,train_indices)
        node_ = tree.nodes
        # Retrain tree
        print("Retraining tree")
        tree.retrain()
    else:
        tree = Tree(dataset,train_indices)
        node_ = tree.nodes
    
    ########### accuracy on training data #################
    pred_classes = np.zeros(len(tree.train_x))
    
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
    
    return

if __name__ == '__main__':
    main()

