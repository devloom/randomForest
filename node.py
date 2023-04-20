import numpy as np
import functools
import operator
import itertools
from numba import njit

@njit
def index(array, item):
    for idx, val in np.ndenumerate(array):
        if val == item:
            return idx

class Node:
    def __init__(self, classes, parent=None, branch=None):
        # information from mother
        self.pred_class = None
        self.class_prob = None
        self.classes_total = classes
        self.pixels = None
        self.parent = parent
        self.branch = branch
        # information about dataset
        self.bags = None
        self.fourD = False
        # centroid calculations
        self.classes_subset = None
        self.n_classes = None
        self.centroids = None
        self.cent_split = None
        # points to daughter nodes
        self.left = None
        self.right = None
        self.depth = None
        # if node is retrained
        self.retrain = False
        # training data
        self.X = None
        self.y = None
        # retraining data
        self.retrain_X = []
        self.retrain_y = []

    def get_X(self):
        # if we call get_X on a leaf, we return the stored array
        if (self.left is None) and (self.right is None):
            return self.X
        else:
            return self.flatten(self.X)

    def get_y(self):
        # if we call get_y on a leaf, we return the stored array
        if (self.left is None) and (self.right is None):
            return self.y
        else:
            return self.flatten(self.y)

    def flatten(self,x):
        result = []
        for el in x:
            if hasattr(el, "__iter__") and not isinstance(el, (np.ndarray, np.generic)):
                result.extend(self.flatten(el))
            else:
                result.append(el)
        return result

    def grow(self,X,y,pixels,bags,retrain,depth=0):
        # number of samples
        num = len(y)

        '''
        ####################### IN PROGRESS ###############
        ## WARNING (gives runtime error BEFORE grow is called in tree)
        # store the class probailities in a dictionary for greater felxibility
        self.class_prob = dict()
        for typ in self.classes_total:
            num_samples_per_class = np.sum(y == typ)
            class_probability = num_samples_per_class/num
            self.class_prob[typ] = class_probability
        self.pred_class = max(self.class_prob, key=self.class_prob.get)
        ####################### IN PROGRESS ###############
        '''

        # find the class probabilites, set as node attributes
        num_samples_per_class = np.array([np.sum(y == i) for i in self.classes_total])
        self.class_prob = num_samples_per_class/num
        # store the class probailities in a dictionary for greater felxibility
        d = dict()

        #DEBUG
        #print("classes total", self.classes_total)
        for typ in self.classes_total:
            d[typ] = self.class_prob[typ]
        self.class_prob = d
        self.pred_class = max(self.class_prob, key=self.class_prob.get)

        ### DEBUG
        #print("for a node of depth", depth)
        #print("num of samples per class", num_samples_per_class)
        #print("class prob:", self.class_prob)
        #print("we predict class", self.pred_class)
        # Create a node, set attributes and find the splitting function
        self.pixels = pixels
        self.bags = bags
        self.depth = depth
        self.retrain = retrain

        # Delete the classes which have no associated samples and take a subset
        new_classes = np.delete(self.classes_total, np.where(num_samples_per_class == 0))
        self.classes_subset = np.random.choice(new_classes,int(np.ceil(np.sqrt(len(new_classes)))),replace=False)
        self.n_classes = len(self.classes_subset)

        X_sub = np.array([X[i] for i in range(len(X)) if (y[i] in self.classes_subset)])
        y_sub = np.array([label for label in y if (label in self.classes_subset)])
        # call splitter
        self.splitter(X_sub, y_sub)


        # From the splitting function & centroids assing the data to go to either the left or right right node
        indices_left = [False]*num
        ## DEBUG
        #print("cent_split in grow:", self.cent_split)
        if self.cent_split is not None:
            # for each feature return the index of the nearest centroid where centroids are calculated for each class in the subclass array of length sqrt(|K|)
            nearest_cent_idx = np.argmin(np.array([[np.linalg.norm(X[i] - self.centroids[k]) for k in range(self.n_classes)] for i in range(num)]),axis=1)
            indices_left = np.array([True if np.any(np.nonzero(self.cent_split == 0)[0] == nearest_cent_idx[j]) else False for j in range(len(nearest_cent_idx))])
            X_left, y_left = X[indices_left], y[indices_left]
            X_right, y_right = X[~indices_left], y[~indices_left]
            # Instantiate and Recursively call grow on the left and right daughters
            self.left = Node(self.classes_total, self, "left")
            self.left.fourD = self.fourD
            (self.left).grow(X_left, y_left, pixels, bags, self.retrain, depth + 1)

            # call grow on the right daughter
            self.right = Node(self.classes_total, self, "right")
            self.right.fourD = self.fourD
            (self.right).grow(X_right, y_right, pixels, bags, self.retrain, depth + 1)

            # which data the node was trained on (needed during retraining)
            if not self.retrain:
                # in this case we assign self.X and self.Y as a list of the daughters
                self.X = list([(self.left).X,(self.right).X])
                self.y = list([(self.left).y,(self.right).y])
        else:
            if not self.retrain:
                # in the leaf case we assign the actual data to self.X and self.y
                self.X = X
                self.y = y

        return

    def splitter(self,X,y):
        if (len(y) <= 5):
            self.cent_split = None
            ## DEBUG
            #print("Reached end condition. self.cent_split:",self.cent_split)
            return

        num_parent = [np.sum(y == i) for i in self.classes_subset]
        best_gini = 1.0 - sum((n / (len(y))) ** 2 for n in num_parent)
        ite = 0

        # Initialize depending on the datastructure we are using. 4D arrays or bag of words
        ## DEBUG
        #print("4D according to node is:", self.fourD)
        if self.fourD:
            cent = np.zeros((self.n_classes,self.pixels,self.pixels,3))
        else:
            cent = np.zeros((self.n_classes,self.bags))

        num_parent = [np.sum(y == i) for i in self.classes_subset]

        # Summing training data of respective centroid class
        for i in range(len(X)):
            cls_idx = index(self.classes_subset,y[i])[0]
            cent[cls_idx] += X[i]
        # Calculate centroid
        self.centroids = np.array([cent[i]/num_parent[i] for i in range(len(self.classes_subset))])
        # Find distance to each centroid and find the closest one
        #print(self.centroids[0])
        cent_distance = np.array([[np.linalg.norm(X[i] - self.centroids[k]) for k in range(self.n_classes)] for i in range(len(X))])
        nearest_cent_idx = np.argmin(cent_distance,axis=1)
        # DEBUG
        #print("cent_distance:", cent_distance)
        #print("closeset:", nearest_cent_idx)

        for i in range(20):
            centroids_split = np.random.randint(2,size=len(self.centroids))
            num_left = [0]*self.n_classes
            num_right = num_parent.copy()
            tot_left = 0
            tot_right = len(y)

            for j in range(len(X)):

                cls_idx = index(self.classes_subset,y[j])[0]
                if (centroids_split[nearest_cent_idx[j]] == 0):
                    num_left[cls_idx] += 1
                    num_right[cls_idx] -= 1
                    tot_left += 1
                    tot_right -= 1
            #calculate gini here
            gini_left = 0.0 if tot_left == 0 else (1.0 - sum((num_left[z]/tot_left)**2 for z in range(len(num_left))))
            gini_right = 0.0 if tot_right == 0 else (1.0 - sum((num_right[z]/tot_right)**2 for z in range(len(num_right))))
            gini = (tot_left*gini_left + tot_right*gini_right)/(tot_left+tot_right)
            ## DEBUG
            #print("does gini improve:", gini < best_gini)
            #print(tot_left, tot_right, gini_right, gini_left)
            if (gini < best_gini):
                best_gini = gini
                self.cent_split = centroids_split
                # DEBUG
                #print("cent_split in splitter:", centroids_split)
        return


    # function to unpack nodes and store them in a list
    def find_daughters(self):
        node_list = [self]
        for node in node_list:
            if node.left is not None:
                node_list.append(node.left)
            if node.right is not None:
                node_list.append(node.right)
        # remove first element, we only want the daughter nodes
        node_list = node_list[1:]
        return node_list

def main():
    return

if __name__ == "__main__":
    main()
