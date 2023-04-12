from node import Node
from dataset import Dataset
from node import Node

import numpy as np
from numba import njit
from tqdm import tqdm

@njit
def index(array, item):
    for idx, val in np.ndenumerate(array):
        if val == item:
            return idx

class Tree():
    def __init__(self,data):
        super().__init__()

        self.max_depth = 3
        self.splittingFunction = 'gini'
        self.classes = np.array(list(set(data.train_y)))

        self.n_classes = len(self.classes)
        #self.n_classes = len(set(data.train_y))
        
        self.nodes = self.grow(data.train_x,data.train_y)

    '''
    def entropy(self,p):
        #### this is informationgain function from lecture 21 slides on decision trees ########
        h = 0
        for i in range(p):
            h += 
        h = -p*np.log2(p) - (1-p)*np.log2(1-p)
        return h
    '''
    
    #determine best split that specifies col #, row #, (r,g,or,b), threshold for each node
    # this still needs to be written
    # we need to discuss best way to go about splitting
    def splitter(self,X,y):
        best_col, best_row, best_rgb, best_thr = None, None, None, None

        if (len(y) <= 1):
            return None,None,None,None

        #num_parent = [np.sum(y == i) for i in range(self.n_classes)]
        num_parent = [np.sum(y == i) for i in self.classes]
        best_gini = 1.0 - sum((n / (len(y))) ** 2 for n in num_parent)
        
        print(best_gini)

        ite = 0
        ###work in progress
        for row in tqdm(range(len(X[0]))):
            #print (row)
            for col in range(len(X[0,:,])):
                for rgb in range(3):
                    for thr in np.arange(0,1,0.1):
                        num_left = [0]*self.n_classes
                        num_right = [0]*self.n_classes
                        tot_left = 0
                        tot_right = 0
                        for i in range(len(X)):
                            #cls_idx = np.argwhere(self.classes==y[i])[0][0]
                            cls_idx = index(self.classes,y[i])[0]
                            if X[i,row,col,rgb] < thr:
                                num_left[cls_idx] += 1
                                tot_left += 1
                            else:
                                #print(y[i])
                                num_right[cls_idx] += 1
                                tot_right += 1

                            
                        #calculate gini here
                        gini_left = 0.0 if tot_left == 0 else (1.0 - sum((num_left[z]/tot_left)**2 for z in range(len(num_left))))
                        gini_right = 0.0 if tot_right == 0 else (1.0 - sum((num_right[z]/tot_right)**2 for z in range(len(num_right))))
                        gini = (tot_left*gini_left + tot_right*gini_right)/(tot_left+tot_right)
                        #######
                        if gini < best_gini:
                            best_gini = gini
                            best_row = row
                            best_col = col
                            best_rgb = rgb
                            best_thr = thr
        
        print (best_gini, best_row,best_col,best_rgb,best_thr)
        return best_row, best_col, best_rgb, best_thr

    
    
    def grow(self,X,y,depth=0):
        #num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes)]
        num_samples_per_class = [np.sum(y == i) for i in self.classes]
        predicted_class = self.classes[np.argmax(num_samples_per_class)]
        print("predicted class: ", predicted_class)
        node = Node(pred_class=predicted_class)
        if depth < self.max_depth:
            rowIdx, colIdx, rgbIdx, thr = self.splitter(X, y)
            indices_left = [False]*len(y)
            if colIdx is not None:
            #if idx is not None:
                #for i in range(len(y)):
                #   if (rowIdx < 0 and np.mean(X[:,rowIdx,colIdx,rgbIdx]) < thr):
                #        indices_left[i] = True
                #        #indices_left = X[:, :, colIdx, rgbIdx] < thr
                #    elif (colIdx < 0 and np.mean(X[:,rowIdx,colIdx,rgbIdx]) < thr):
                #        indices_left[i] = True
                #        #indices_left = X[:, :, colIdx, rgbIdx] < thr
                #indices_left = np.array(indices_left)
                indices_left = X[:,rowIdx,colIdx,rgbIdx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.threshold = thr
                print("Splitting tree at level ", depth)
                node.left = self.grow(X_left, y_left, depth + 1)
                node.right = self.grow(X_right, y_right, depth + 1)
        return node

    def print_leaves(self,node,leaf):
        if node.left == None:  
            print ("leaf: ", leaf,"predicted class: ", node.pred_class)
        else:
            print_leaves(node.left,leaf-1)
            print_leaves(node.right,leaf-2)
            
            

if __name__ == '__main__':
    dataset = Dataset()

    #print(np.array(set(dataset.train_y)))
    #thr = 0.5
    #X = np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])/9
    #print (X[:, 0] < thr)

    tree = Tree(dataset)
    node_ = tree.nodes
    tree.print_leaves(node_,2**(tree.max_depth-1))

    pred_classes = np.zeros(len(dataset.train_x))
    print("here")
    '''
    #for i in range(len(dataset.train_x)):
        #test_img = dataset.train_x[i]
        
        #while node_.left:

            #if test_img[node_.row_index,node_.col_index, node_.rgb_index] < node_.threshold:
            #    node_ = node_.left
            #else:
            #    node_ = node_.right
        #pred_classes[i] = node_.pred_class

    print(dataset.train_y)
    print(pred_classes)
    num = np.sum([1 if dataset.train_y[i] == pred_classes[i] else 0 for i in range(len(pred_classes))])
    print("accuracy: ", num/len(pred_classes))
    '''