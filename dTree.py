class Tree():
	def __init__(self):
        super().__init__()

        self.numTrees = 100
        self.trees = []*numTrees
        self.splittingFunction = 'gini'
        self.createTrees()


    def setSplittingFunction(self,split):
    	self.splittingFunction = split
