class Tree():
    def __init__(self):
        super().__init__()

        self.depth = 5
        self.splittingFunction = 'gini'
        self.nodes = [()]
        #self.createTrees()


    def setSplittingFunction(self,split):
        self.splittingFunction = split

    def node(x,xVal):
        if x < xVal:
            return True
        else:
            return False


    def grow():
        infoGain = 0
        if (self.splittingFunction == 'gini'):
            #### find best first split
            for i in range(0,1,0.1):
                if (self.node(x_mean[]) 

