from node import Node

class Tree():
    def __init__(self,dataset):
        super().__init__()

        self.depth = 5
        self.splittingFunction = 'gini'
        self.n_classes = len(set(dataset.train_y))
        self.nodes = self.grow(dataset.train_x,dataset.train_y)

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
    def splitter(self,X,y):
        best_col = -1
        best_row = -1
        best_rgb = 0
        best_thr = 0.5
        return best_col, best_row, best_rgb, best_thr

    
    
    def grow(self,X,y,depth=0):
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes)]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(pred_class=predicted_class)
        if depth < self.max_depth:
            rowIdx, colIdx, rgbIdx, thr = self.splitter(X, y)
            indices_left = [False]*len(y)
            if idx is not None:
                for i in range(len(y)):
                    img = dataset.imgNumpy(i)
                    if (rowIdx < 0 and np.mean(img[:,colIdx,rgbIdx]) < thr):
                        indices_left[i] = True
                        #indices_left = X[:, :, colIdx, rgbIdx] < thr
                    elif (colIdx < 0 and np.mean(img[rowIdx,:,rgbIdx]) < thr):
                        indices_left[i] = True
                        #indices_left = X[:, :, colIdx, rgbIdx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.threshold = thr
                node.left = self.grow(X_left, y_left, depth + 1)
                node.right = self.grow(X_right, y_right, depth + 1)
        return node
            
            

