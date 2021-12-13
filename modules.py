import torch
import torchvision.transforms as transforms
import torchvision
import sklearn
from sklearn import tree
from sklearn import ensemble
import utils
import numpy as np
from joblib import dump, load


class HanpiDecisionTree():
    """
        A self defined decision tree based on sklearn.tree.DecisionTreeClassifier.
        Initialization: input a decision tree "DT"
    """
    def __init__(self, DT):
        self.DT = DT # The main decision tree
        self.depth = self.DT.get_depth() # Depth of this tree
        self.features = utils.find_features(self.DT) # All features used in this tree
        self.selected_features = [] # A subset of self.features. The selected features ar used to train a compact tree
        self.ori_map = utils.map_CAM(self.DT) # Find the original mapping of this tree
    
    def select_features(self, n_features, method):
        """
            randomly select the number "n_features" of features to form a subset "self.selected_features" of "self.features"
        """
        if n_features > len(self.features): # if the number of feature needed is more than the actual number of features used.
            self.selected_features = self.features
        else:
            if method == "random":
                self.selected_features = np.random.choice(self.features, n_features, replace=False)
            elif method == "size":
                size = np.sum(self.ori_map, axis = 0)
                res = np.argsort(size)[::-1]
                self.selected_features = res[:n_features]
            else:
                raise NotImplementedError
    
    def generate_input(self, X):
        """
            Generate a comptact input vector that only consits of the selected features
        """
        T = []
        for i in self.selected_features:
            line = X[:,i]
            T.append(line)
        T = np.array(T)
        T = T.transpose()
        return T

    def predict(self, X):
        """
            The inference of this decision tree.
        """
        if len(self.selected_features) == 0:
            return self.DT.predict(X)
        else:
            XX = self.generate_input(X)
            return self.DT.predict(XX)
    
    def fit(self, X, Y):
        """
            The training of this decision tree.
        """
        XX = self.generate_input(X)
        self.DT.fit(XX, Y)

class MoRandomForest():
    """
        A self defined random forest based on sklearn.ensemble.RandomForestClassifier.
        Initialization: input the estimators of a random forest: sklearn.ensemble.RandomForestClassifier.estimators_
    """
    def __init__(self, estimators):
        self.estimators_ = [] # the estimators
        self.transform(estimators) # transform all the decision tree in the random forest to HanpiDecisionTree
        
    def transform(self, estimators):
        """
            transform all the decision tree in the random forest to HanpiDecisionTree
        """
        for i in range(len(estimators)):
            self.estimators_.append(HanpiDecisionTree(estimators[i]))

    def predict(self, X):
        """
            The inference of this random forest.
            The ensemble may not be perfect and may be slow.
        """
        results = []
        predicted = []
        for est in self.estimators_:
            pred = est.predict(X)
            results.append(pred)
        results = np.array(results)
        for i in range(len(results[0])):
            this = results[:,i]
            res = np.bincount(this.astype(np.int32)).argmax()
            predicted.append(res)
        predicted = np.array(predicted)
        return predicted