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
    def __init__(self,data,firstIdx = 0,lastIdx = 50000,test=False):
        super().__init__()

        self.max_depth = 10
        self.pixels = data.pixels
        

        #indices = sorted(np.array([i for i in range(len(data.train_dataset["image"]))]),key=lambda k:random.random())
        indices = sorted(np.array([i for i in range(len(data.train_dataset["img"]))]),key=lambda k:random.random())
        #print(indices)


        indices_sub = np.array(indices[firstIdx:lastIdx])
        #indices_sub = indices[firstIdx:lastIdx]
        
        #print(type(indices_sub))
        #self.train_img = [image.convert("RGB").resize((data.pixels,data.pixels)) for image in data.train_dataset["image"]]
        #self.train_img = [data.train_dataset[i.item()]["image"].convert("RGB").resize((data.pixels,data.pixels)) for i in indices_sub]
        self.train_img = [data.train_dataset[i.item()]["img"].convert("RGB").resize((data.pixels,data.pixels)) for i in indices_sub]
        #print("here")
        self.train_x = np.array([data.imgNumpy(image) for image in self.train_img])
        #print("here")
        #self.train_y = np.array(data.train_dataset['labels'])[indices_sub.astype(int)]
        self.train_y = np.array(data.train_dataset['label'])[indices_sub.astype(int)]
        #print("here")

        self.classes = np.array(list(set(self.train_y)))

        self.n_classes = len(self.classes)
        #self.n_classes = len(set(data.train_y))

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

    #determine best split that specifies col #, row #, (r,g,or,b), threshold for each node
    # this still needs to be written
    # we need to discuss best way to go about splitting


    ######### NCMC ################
    #### compute centroid for each class 
    #### assign centroid randomly either a -1 (left) or 1 (right)
    #### optimize how we randomly assign centroids by optimizing information gain


    def splitter(self,X,y):
        #best_col, best_row, best_rgb, best_thr = None, None, None, None
        best_cent_split,nearest_cent_ind,centroids = None,None,None

        if (len(y) <= 5):
            #print("here")
            return None,None,None


        #num_parent = [np.sum(y == i) for i in range(self.n_classes)]
        num_parent = [np.sum(y == i) for i in self.classes]
        best_gini = 1.0 - sum((n / (len(y))) ** 2 for n in num_parent)
        
        #print("best gini: ", best_gini)

        ite = 0

        #compute centroid
        cent = np.zeros((self.n_classes,self.pixels,self.pixels,3))
        num_parent = [np.sum(y == i) for i in self.classes]

        for i in range(len(X)):
            cls_idx = index(self.classes,y[i])[0]
            cent[cls_idx] += X[i]
        centroids = np.array([cent[i]/num_parent[i] for i in range(len(self.classes))])
        #print(centroids)

        #plt.imshow(centroids[1])
        #plt.show()


        nearest_cent_idx = np.argmin(np.array([[np.linalg.norm(X[i] - centroids[k]) for k in range(self.n_classes)] for i in range(len(X))]),axis=1)
        #print(nearest_cent_idx)
        #print(np.array([[np.linalg.norm(X[i] - self.centroids[k]) for k in range(self.n_classes)] for i in range(len(X))]))

        for i in range(30):
            centroids_split = np.random.randint(2,size=len(centroids))
            num_left = [0]*self.n_classes
            num_right = num_parent.copy()
            tot_left = 0
            tot_right = len(y)
            
            for j in range(len(X)):
                cls_idx = index(self.classes,y[j])[0]
                if (centroids_split[nearest_cent_idx[j]] == 0):
                    num_left[cls_idx] += 1
                    num_right[cls_idx] -= 1
                    tot_left += 1
                    tot_right -= 1
            #calculate gini here
            gini_left = 0.0 if tot_left == 0 else (1.0 - sum((num_left[z]/tot_left)**2 for z in range(len(num_left))))
            gini_right = 0.0 if tot_right == 0 else (1.0 - sum((num_right[z]/tot_right)**2 for z in range(len(num_right))))
            gini = (tot_left*gini_left + tot_right*gini_right)/(tot_left+tot_right)
            #print('lft: ', tot_left)
            #print('rht: ',tot_right)
            #print('gini: ', gini)
            #print('split',centroids_split)
            if (gini < best_gini):
                best_gini = gini
                best_cent_split = centroids_split
                #nearest_cent_ind = nearest_cent_idx

        #print(best_cent_split, best_gini)
        return best_cent_split, nearest_cent_idx, centroids



        


        #centroids[index(self.classes,y[i])[0]]

        #np.sum(X[,:])






        #centroid[np.mean(X,axis=(0,1))]


        ###work in progress
        '''
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
        
        #print (best_gini, best_row,best_col,best_rgb,best_thr)
        return best_row, best_col, best_rgb, best_thr
        '''

    
    
    def grow(self,X,y,depth=0):
        #num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes)]
        num_samples_per_class = [np.sum(y == i) for i in self.classes]

        class_probability = [np.sum(y == i)/len(y) for i in self.classes]
        predicted_class = self.classes[np.argmax(num_samples_per_class)]
        #print("Tree at depth ", depth)
        #print("predicted class: ", predicted_class)
        node = Node(pred_class=predicted_class,class_prob=class_probability)

        if depth < self.max_depth:
            #rowIdx, colIdx, rgbIdx, thr = self.splitter(X, y)
            bestCentSplit, nearestCentIdx, nodeCentroids = self.splitter(X, y)
            # = self.splitter(X, y)
            indices_left = [False]*len(y)
            if bestCentSplit is not None:
            #if idx is not None:
                #for i in range(len(y)):
                #   if (rowIdx < 0 and np.mean(X[:,rowIdx,colIdx,rgbIdx]) < thr):
                #        indices_left[i] = True
                #        #indices_left = X[:, :, colIdx, rgbIdx] < thr
                #    elif (colIdx < 0 and np.mean(X[:,rowIdx,colIdx,rgbIdx]) < thr):
                #        indicesindices_left_left[i] = True
                #        #indices_left = X[:, :, colIdx, rgbIdx] < thr
                #indices_left = np.array(indices_left)
                ##indices_left = X[:,rowIdx,colIdx,rgbIdx] < thr
                #bestCentSplit = np.array([1,0,0])
                #print(bestCentSplit == 0)
                #print(np.nonzero(bestCentSplit == 0)[0])
                #if nearestCentIdx[i]:
                #    indices_left[i] = True
                #else:
                #    indices_left[i] = False
                indices_left = np.array([True if np.any(np.nonzero(bestCentSplit == 0)[0] == nearestCentIdx[j]) else False for j in range(len(nearestCentIdx))])
                #indices_left = nearestCentIdx == 0
                #indices_left = np.array(indices_left)
                #print(indices_left)
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.cent_split = bestCentSplit
                node.centroids = nodeCentroids
                ##node.row_index = rowIdx
                ##node.col_index = colIdx
                ##node.rgb_index = rgbIdx
                ##node.threshold = thr
                
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

    #print(np.array(set(dataset.train_y)))
    #thr = 0.5
    #X = np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])/9
    #print (X[:, 0] < thr)
    #print(dataset.train_x[0,:,:])

    
    tree = Tree(dataset)
    node_ = tree.nodes
    #tree.print_leaves(node_,2**(tree.max_depth-1))
    tree.print_leaves(node_)

    
    pred_classes = np.zeros(len(tree.train_x))
    print("here")
    
    for i in range(len(tree.train_x)):
    #for i in range(1950,2050,1):
        #print(i)
        node_ = tree.nodes
        test_img = tree.train_x[i]
        
        while node_.left:
            nearest_cent = np.argmin(np.array([np.linalg.norm(tree.train_x[i] - node_.centroids[k]) for k in range(tree.n_classes)]))
            if (i == 2):
                print(nearest_cent)
                print(node_.cent_split[nearest_cent])
            if (node_.cent_split[nearest_cent] == 0):
                node_ = node_.left
            else:
                node_ = node_.right
        if (i == 2):
            print("pred: ", node_.pred_class)
        pred_classes[i] = node_.pred_class

    print(tree.train_y[0:100])
    print(pred_classes[0:100])
    
    num = np.sum([1 if tree.train_y[i] == pred_classes[i] else 0 for i in range(len(pred_classes))])
    print("accuracy: ", num/len(pred_classes))

