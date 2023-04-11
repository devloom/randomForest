class Node:
	def __init__(self, pred_class):
		self.pred_class = pred_class
		self.row_index = 0
		self.col_index = 0
		self.rgb_index = 0
		self.threshold = 0
		self.left = None
		self.right = None