class Node:
	def __init__(self, gini, num, num_samples_per_class, pred_class):
		self.gini = gini
		self.num = num
		self.num_per_class = num_per_class
		self.pred_class = pred_class
		self.row_index = 0
		self.col_index = 0
		self.rgb_index = 0
		self.threshold = 0
		self.left = None
		self.right = None