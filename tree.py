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
            # we are not retraining during initialization
            retrain = False
            self.root = Node(self.classes)
            ## DEBUG
            #print("X:", self.train_x)
            #print("y:", self.train_y)
            #print("pixels:", self.pixels)
            #print("retrain:", retrain)
            (self.root).grow(X=self.train_x,y=self.train_y,pixels=self.pixels,retrain=retrain)
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
        new_X = np.array([data.imgNumpy(image) for image in new_train_img])
        new_y = np.array(data.second_train['label'])[self.indices.astype(int)]
        # find daughters of the root node
        node_list = (self.root).find_daughters()
        # DEBUG 
        #print("node list", len(node_list))

        # uniform probability of retraining each node (currently at 0%)
        retrain_nodes = [node for node in node_list if (np.random.uniform() >= 0.99)]
        print("retraining selection including daughters", len(retrain_nodes))
        # remove the daughter nodes to avoid double retaining nodes
        for node in retrain_nodes:
            node_daughters = node.find_daughters()
            retrain_nodes = [node for node in retrain_nodes if node not in node_daughters]
            # Delete the nodes which will be trained over in retraining
            for condemned in node_daughters:
                del condemned
        # DEBUG    
        print("retraining selection final", len(retrain_nodes))
        i = 0
        for node in retrain_nodes:
            # DEBUG
            print("Retrained node:", i)
            i += 1

            # get the list of arrys on which node was trained
            list_X = node.get_X()
            list_y = node.get_y()
            # if node is a leaf, we concatenate list_X (which is an array) with the new training data
            if isinstance(list_X, (np.ndarray, np.generic)):
                comb_train_x = np.concatenate((list_X,new_X))
                comb_train_y = np.concatenate((list_y,new_y))
            # if the node is not a leaf, we append the new training data to the list of arrays on which node was trained
            else:
                list_X.append(new_X)
                # if node is not a leaf, list_X is a list of arrays
                comb_train_x = np.concatenate(list_X)
                # get the list of y data that node was trained on and append the new training data
                list_y.append(new_y)
                comb_train_y = np.concatenate(list_y)
                          
            # Reset self.classes for the larger dataset
            self.classes = np.array(list(set(comb_train_y)))
            self.n_classes = len(self.classes)
            # We reassign the retrained node to the position it occupied in its parent
            if node.branch == "left":
                new_node = Node(self.classes, node.parent, "left")
                (node.parent).left = new_node
                new_node.grow(comb_train_x, comb_train_y, self.pixels, depth=node.depth, retrain=True)
            elif node.branch == "right":
                new_node = Node(self.classes, node.parent, "right")
                (node.parent).right = new_node
                new_node.grow(comb_train_x, comb_train_y, self.pixels, depth=node.depth, retrain=True)
            else:
                print("node.branch was undefined")
                break


def main(increment=True):
    # Initialize dataset
    dataset = Dataset()

    # arbitrary indicies determine how large the training dataset is
    train_indices = np.array([i for i in range(0,10000,1)])


    if increment:
        # split the data for incrmental learning
        init_classes = 5
        dataset.split_data(init_classes)
        # Train tree on initial data and calculate nodes (node_ is the root node)
        print("Training tree on original data")
        tree = Tree(dataset,train_indices)
        node_ = tree.root
        print(node_)
        # Retrain tree
        print("Retraining tree")
        tree.retrain()
    else:
        tree = Tree(dataset,train_indices)
        node_ = tree.root
    
    ########### accuracy on training data #################
    pred_classes = np.zeros(len(tree.train_x))
    
    for i in range(len(train_indices)):
    #for i in range(1950,2050,1):
        print(i)
        node_ = tree.root
        #test_img = tree.train_x[i]
        
        while node_.left or node_.right:
            nearest_cent = np.argmin(np.array([np.linalg.norm(tree.train_x[i] - node_.centroids[k]) for k in range(node_.n_classes)]))
            print("left",node_.left)
            print("right",node_.right)
            #if (i == 2):
            #    print(nearest_cent)
            print(node_.cent_split[nearest_cent])
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
        node_ = tree.root
        test_img = dataset.test_x[i]
        
        while node_.left or node_.right:
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

