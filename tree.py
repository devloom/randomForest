class Tree():
    def __init__(self):
        super().__init__()

        self.depth = 5
        self.splittingFunction = 'gini'
        self.nodes = [[]]
        #self.createTrees()


    def setSplittingFunction(self,split):
        self.splittingFunction = split

    def node(x_train,col,xVal):
        if x_train[col] < xVal:
            return True
        else:
            return False


    def grow():
        
        if (self.splittingFunction == 'gini'):
            infoGain = 0
            #### find best first split
            for i in range(0,1,0.1):

