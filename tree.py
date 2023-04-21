from node import Node
from dataset import Dataset
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
        self.bags = data.bags
        ## DEPRECATED 
        # Index data should come from the split, and not passed as an initializing argument
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
            self.train_X = np.array([data.imgNumpy(image) for image in self.train_img])
            self.train_y = np.array(dataset_head['label'])
        else:    
            # Previous method of splitting up training data with no streamed data
            # Image resizing for training data occurs in tree
            ######DEPRECATED############
            #self.train_img = [data.train_dataset[i.item()]["img"].convert("RGB").resize((data.pixels,data.pixels)) for i in indices]
            #self.train_x = np.array([data.imgNumpy(image) for image in self.train_img])
            #self.train_y = np.array(data.train_dataset['label'])[indices.astype(int)]
            ##########################
            ###### BAG OF WORDS ########
            self.train_X = data.train_X[indices]
            self.train_y = data.train_y[indices]

        
        # for use in grow
        self.classes = np.array(list(set(self.train_y)))
        self.n_classes = len(self.classes)

        ## IN PROGRESS (Ability to save trees and then reload)
        ### test = false means we need to train the tree
        ### test = true means the tree has already been trained and we read in hyperparameters from file
        if (test == False):
            # we are not retraining during initialization
            retrain = False
            self.root = Node(self.classes)
            self.root.fourD = self.data.fourD
            (self.root).grow(X=self.train_X,y=self.train_y,pixels=self.pixels,bags=self.bags,retrain=retrain)
        else:
            print("Reading in tree:")
            #### code here to read in node structure of tree
    
    '''
    ## IN PROGRESS (Entropy function for splitting function, currently using gini scores)
    def entropy(self,p):
        #### this is informationgain function from lecture 21 slides on decision trees ########
        h = 0
        for i in range(p):
            h += 
        h = -p*np.log2(p) - (1-p)*np.log2(1-p)
        return h
    '''

    def print_leaves(self,node):
        if node.left == None:  
            print ("predicted class: ", node.pred_class)
        else:
            self.print_leaves(node.left)
            self.print_leaves(node.right)

    def retrain(self, indices):
        # import the data
        data = self.data
        # We combine the new data with the old training data for retraining
        ############# We can't recreate new features with SIFT here, it takes too long, should just use what is already created in dataset
        ############# probably should make split data split the actually train features, not the datasets
        '''
        new_train_img = [data.second_train[i.item()]["img"].convert("RGB").resize((self.pixels,self.pixels)) for i in indices]
        new_X = np.array([data.imgNumpy(image) for image in new_train_img])
        new_y = np.array(data.second_train['label'])[indices.astype(int)]
        '''
        ## DEBUG
        new_X = data.second_train_X[indices]
        new_y = data.second_train_y[indices]

        # Reset self.classes for the larger dataset
        #self.classes = np.concatenate((self.classes,np.array(list(set(data.second_train['label'])))))
        self.classes = np.concatenate((self.classes,np.array(list(set(data.second_train_y)))))
        self.n_classes = len(self.classes)

        # sort the additional training data into the leaves of the original tree
        self.sort(new_X, new_y)

        # find daughters of the root node
        node_list = (self.root).find_daughters()

        # uniform probability of retraining each node (currently at 40%)
        retrain_nodes = [node for node in node_list if (np.random.uniform() >= 0.80)]
        ## DEBUG
        #print("retraining selection including daughters", len(retrain_nodes))

        for node in retrain_nodes:
            # remove the daughter nodes form the retrain list to avoid double retaining nodes
            node_daughters = node.find_daughters()
            retrain_nodes = [node for node in retrain_nodes if node not in node_daughters]
        ## DEBUG
        #print("New length of retrain nodes is:", len(retrain_nodes))

        # DEBUG    
        i = 0
        for node in retrain_nodes:
            # DEBUG
            #print("Retrained node:", i)
            i += 1

            # find all retrain data from the daughters 
            node_daughters = node.find_daughters()
            raw_new_X = []
            raw_new_y = []
            for daughter in node_daughters:
                raw_new_X.append(daughter.retrain_X)
                raw_new_y.append(daughter.retrain_y)
            # clean out empty lists from the retrain data
            new_X = [ele for ele in raw_new_X if ele != []]
            new_y = [ele for ele in raw_new_y if ele != []]

            # get the list of arrays on which node was trained initially
            list_X = node.get_X()
            list_y = node.get_y()

            # if node is a leaf, we append list_X (which is an array) to the new training data
            if isinstance(list_X, (np.ndarray, np.generic)):
                new_X.append(list_X)
                new_y.append(list_y)
                # Then we concatenate
                comb_train_X = np.concatenate(new_X) 
                comb_train_y = np.concatenate(new_y)
            # if the node is not a leaf, we append the new training data to the list of arrays on which node was trained
            else:
                # if node is not a leaf, we extend list_X (which is a list of arrays) with the retraining data
                list_X.extend(new_X)
                list_y.extend(new_y)
                # Then we concatenate
                comb_train_X = np.concatenate(list_X)
                comb_train_y = np.concatenate(list_y)
                          
            # We reassign the retrained node to the position it occupied in its parent, and continue to grow the tree on the combined original and retraining data
            if node.branch == "left":
                new_node = Node(self.classes, node.parent, "left")
                (node.parent).left = new_node
                new_node.grow(comb_train_X, comb_train_y, self.pixels, depth=node.depth, retrain=True, bags=self.bags)
            elif node.branch == "right":
                new_node = Node(self.classes, node.parent, "right")
                (node.parent).right = new_node
                new_node.grow(comb_train_X, comb_train_y, self.pixels, depth=node.depth, retrain=True, bags=self.bags)
            else:
                print("node.branch was undefined")
                break
        return

    def sort(self, X, y=[], testing=False):
        # if we are testing using this algorithim, we want to return the predicted classes
        if testing:
            pred_classes = np.zeros(len(X))
            class_probs = []
        # loop over all the elements and either sort them to their leaves or find the leaf's predicted class
        for i in range(len(X)):
            node_ = self.root
            while node_.left or node_.right:
                nearest_cent = np.argmin(np.array([np.linalg.norm(X[i] - node_.centroids[k]) for k in range(node_.n_classes)]))
                if (node_.cent_split[nearest_cent] == 0):
                    node_ = node_.left
                else:
                    node_ = node_.right
            if not testing:
                node_.retrain_X.append(X[i])
                node_.retrain_y.append(y[i])
            if testing:
                pred_classes[i] = node_.pred_class
                class_probs.append(node_.class_prob)
        
        # return the predicted classes if we are testing
        if testing:
            # DEBUG
            #print("pred_classes in sort:", pred_classes)
            #print("class_probs in sort:", class_probs)
            return pred_classes, class_probs


def main(increment=True):
    # Initialize dataset

    #What percentage of the available training dataset do you want to use for training? Enter [0,1]
    train_percent = 0.1
    #What percentage of the available testing dataset do you want to use for testing? Enter [0,1]
    test_percent = 0.1

    dataset = Dataset(train_pct=train_percent,test_pct=test_percent)
    #dataset = Dataset()

    # arbitrary indices determine how large the training dataset is
    # DEPRECATED (The indices should be set by the training data, not via the instantiator)
    
    #train_indices = np.array([i for i in range(0,25000,1)])


    if increment:
        # split the data for incrmental learning
        init_classes = 5
        dataset.split_data(init_classes)

        train_length = len(dataset.train_X)
        train_indices = sorted(np.array([i for i in range(train_length)]),key=lambda k:random.random())
        # Train tree on initial data and calculate nodes (node_ is the root node)
        print("Training tree on original data")
        tree = Tree(dataset,train_indices)
        node_ = tree.root
        # Retrain tree
        print("Retraining tree")
        ## DEPRECATED (This is only for our current dataset. In the future only run forest.py)
        #retrain_indices = np.array([i for i in range(0,25000,1)])
        retrain_length = len(dataset.second_train_X)
        retrain_indices = sorted(np.array([i for i in range(retrain_length)]),key=lambda k:random.random())
        tree.retrain(retrain_indices)
    else:
        train_length = len(dataset.train_X)
        train_indices = sorted(np.array([i for i in range(train_length)]),key=lambda k:random.random())
        tree = Tree(dataset,train_indices)
        node_ = tree.root
    
    ########### accuracy on training data #################
    pred_classes, class_probs = tree.sort(tree.train_X,testing=True)

    print(tree.train_y[0:100])
    print(pred_classes[0:100])

    num = np.sum([1 if tree.train_y[i] == pred_classes[i] else 0 for i in range(len(pred_classes))])
    
    print("Validation accuracy: ", num/len(pred_classes))
    
    ########### accuracy on test data #################
    pred_classes, class_probs = tree.sort(dataset.test_X,testing=True)

    print(dataset.test_y[0:100])
    print(pred_classes[0:100])

    num = np.sum([1 if dataset.test_y[i] == pred_classes[i] else 0 for i in range(len(pred_classes))])
    
    print("Test accuracy: ", num/len(pred_classes))
    
    return

if __name__ == '__main__':
    main()

