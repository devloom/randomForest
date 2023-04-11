from node import Node
from dataset import Dataset
from node import Node

import numpy as np

class Tree():
    def __init__(self,data):
        super().__init__()

        self.max_depth = 2
        self.splittingFunction = 'gini'
        self.n_classes = len(set(data.train_y))
        self.nodes = self.grow(data.train_x,data.train_y)

        #self.createTrees()


    def setSplittingFunction(self,split):
        self.splittingFunction = split

    def gini(self,p_classes):
        gini = 0
        for i in range(len(p_classes)):
            gini += p_classes[i]**2
        return 1 - gini

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
        best_col = -1
        best_row = -1
        best_row = 1
        best_rgb = 0
        best_thr = 0.5
        return best_col, best_row, best_rgb, best_thr

    
    
    def grow(self,X,y,depth=0):
        y = np.array(y)
        print(type(X[0]))
        X = np.array(X,dtype='JpegImageFile')
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes)]
        predicted_class = np.argmax(num_samples_per_class)
        #print(num_samples_per_class)
        #print(predicted_class)
        node = Node(pred_class=predicted_class)
        if depth < self.max_depth:
            rowIdx, colIdx, rgbIdx, thr = self.splitter(X, y)
            indices_left = [False]*len(y)
            if (colIdx > 0 or rowIdx > 0):
            #if idx is not None:
                for i in range(len(y)):
                    img = dataset.imgNumpy(i)
                    if (rowIdx < 0 and np.mean(img[:,colIdx,rgbIdx]) < thr):
                        indices_left[i] = True
                        #indices_left = X[:, :, colIdx, rgbIdx] < thr
                    elif (colIdx < 0 and np.mean(img[rowIdx,:,rgbIdx]) < thr):
                        indices_left[i] = True
                        #indices_left = X[:, :, colIdx, rgbIdx] < thr
                indices_left = np.array(indices_left)
                
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.threshold = thr
                node.left = self.grow(X_left, y_left, depth + 1)
                node.right = self.grow(X_right, y_right, depth + 1)
        return node
            
            

if __name__ == '__main__':
    dataset = Dataset()
    #thr = 0.5
    #X = np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])/9
    #print (X[:, 0] < thr)

    tree = Tree(dataset)