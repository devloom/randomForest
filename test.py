from node import Node
from dataset import Dataset
from node import Node
from tree import Tree

import numpy as np
from numba import njit
from tqdm import tqdm

if __name__ == '__main__':
    dataset = Dataset()

    ### Read in already trained tree parameters from file 
    #tree = Tree(dataset,True)
    tree = Tree(dataset)
    node_ = tree.nodes
    #tree.print_leaves(node_,2**(tree.max_depth-1))
    #tree.print_leaves(node_)

    pred_classes = np.zeros(len(dataset.test_x))
    print("here")
    
    for i in range(len(dataset.test_x)):
    #for i in range(1950,2050,1):
        #print(i)
        node_ = tree.nodes
        test_img = dataset.test_x[i]
        while node_.left:
            #print(node_.row_index,node_.col_index, node_.rgb_index,node_.threshold)
            if test_img[node_.row_index,node_.col_index, node_.rgb_index] < node_.threshold:
                node_ = node_.left
            else:
                node_ = node_.right
        pred_classes[i] = node_.pred_class

    print(dataset.test_y[0:100])
    print(pred_classes[0:100])
    
    num = np.sum([1 if dataset.test_y[i] == pred_classes[i] else 0 for i in range(len(pred_classes))])
    print("accuracy: ", num/len(pred_classes))
    