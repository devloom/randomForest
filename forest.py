import math
import sklearn

from dTree import Tree


class Forest():
	def __init__(self):
        super().__init__()

        self.numTrees = 100
        self.trees = []*numTrees
        self.splittingFunction = 'gini'
        self.createTrees()


    def createTrees(self):
        for i in range(numTrees):
            tree = Tree()
            tree.setSplittingFunction(self.splittingFunction)

            self.trees[i] = tree

        	