import numpy as np
from numba import njit

@njit
def index(array, item):
	for idx, val in np.ndenumerate(array):
		if val == item:
			return idx

class Node:
	def __init__(self, pred_class,class_prob,classes,pixels):
		self.pred_class = pred_class
		self.class_prob = class_prob
		self.pixels = pixels
		self.classes = classes
		self.n_classes = len(self.classes)
		self.centroids = None
		self.cent_split = None
		#self.row_index = 0
		#self.col_index = 0
		#self.rgb_index = 0
		#self.threshold = 0
		self.left = None
		self.right = None


	def splitter(self,X,y):
		best_cent_split,nearest_cent_ind,centroids = None,None,None
        
		if (len(y) <= 5):
			#print("here")
			return None,None,None


        
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
		nearest_cent_idx = np.argmin(np.array([[np.linalg.norm(X[i] - centroids[k]) for k in range(self.n_classes)] for i in range(len(X))]),axis=1)

		for i in range(50):
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
		    if (gini < best_gini):
		        best_gini = gini
		        best_cent_split = centroids_split
		        
		return best_cent_split, nearest_cent_idx, centroids