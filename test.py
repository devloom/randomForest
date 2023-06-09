from node import Node
from dataset import Dataset
from node import Node
from tree import Tree

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from tqdm import tqdm

if __name__ == '__main__':
    dataset = Dataset()
    tree = Tree(dataset)
    node_ = tree.nodes

    
    pred_classes = np.zeros(len(dataset.test_x))
    print("here")
    
    for i in range(len(dataset.test_x)):
        node_ = tree.nodes
        test_img = dataset.test_x[i]
        
        while node_.left:
            nearest_cent = np.argmin(np.array([np.linalg.norm(dataset.test_x[i] - node_.centroids[k]) for k in range(tree.n_classes)]))

            if (i == 3):
                print(np.array([np.linalg.norm(dataset.test_x[i] - node_.centroids[k]) for k in range(tree.n_classes)]))
                print("Nearest cent: ", nearest_cent)
                print(node_.cent_split[nearest_cent])
            if (node_.cent_split[nearest_cent] == 0):
                node_ = node_.left
            else:
                node_ = node_.right
                
        if (i == 3):
            print("pred: ", node_.pred_class)
        pred_classes[i] = node_.pred_class

    print(dataset.test_y[750:1250])
    print(pred_classes[750:1250])
    
    num = np.sum([1 if dataset.test_y[i] == pred_classes[i] else 0 for i in range(len(pred_classes))])
    print("accuracy: ", num/len(pred_classes))

    plt.imshow(dataset.test_x[4])
    plt.show()
