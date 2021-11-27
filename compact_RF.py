import torch
import torchvision.transforms as transforms
import torchvision
import sklearn
from sklearn import tree
from sklearn import ensemble
from matplotlib import pyplot as plt
import utils
import numpy as np
from joblib import dump, load
from tqdm import tqdm

if __name__ == "__main__":
    BS = 50000
    trainset = torchvision.datasets.MNIST(root='~/Private/data', train=True,
                                            download=False, transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BS,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='~/Private/data', train=False,
                                        download=False, transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=BS,
                                                shuffle=False, num_workers=2)

    # clf = ensemble.RandomForestClassifier(max_depth=5)
    # # clf = tree.DecisionTreeClassifier()
    # for images, labels in trainloader:
    #     clf.fit(images.view(images.size(0),-1).numpy(), labels.numpy())
    # dump(clf, 'trained_RandomForest.joblib') 
    clf = load('trained_RandomForest.joblib') 

    # for images, labels in testloader:
    #     print(clf.score(images.view(images.size(0),-1).numpy(), labels.numpy()))

    ttt = clf.estimators_
    sparsity = []
    for tt in ttt:
        sp, table = utils.find_sparsity(tt)
        sparsity.append(sp)
    print(np.mean(sparsity))

    class HanpiDecisionTree():
        def __init__(self, DT):
            self.DT = DT
            self.depth = self.DT.get_depth()
            self.features = utils.find_features(self.DT)
            self.selected_features = []
        
        def select_features(self):
            if self.depth > len(self.features):
                self.selected_features = self.features
            else:
                self.selected_features = np.random.choice(self.features, self.depth, replace=False)
        
        def generate_input(self, X):
            T = []
            for i in self.selected_features:
                line = X[:,i]
                T.append(line)
            T = np.array(T)
            T = T.transpose()
            return T

        def predict(self, X):
            if len(self.selected_features) == 0:
                return self.DT.predict(X)
            else:
                XX = self.generate_input(X)
                return self.DT.predict(XX)
        
        def fit(self, X, Y):
            XX = self.generate_input(X)
            self.DT.fit(XX, Y)

    class MoRandomForest():
        def __init__(self, estimators):
            self.estimators_ = []
            self.transform(estimators)
            
        
        def transform(self, estimators):
            for i in range(len(estimators)):
                self.estimators_.append(HanpiDecisionTree(estimators[i]))

        def predict(self, X):
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

    clf = load('trained_RandomForest.joblib') 
    m = MoRandomForest(clf.estimators_)
    for images, labels in testloader:
        pred = m.predict(images.view(images.size(0),-1).numpy())
        print((pred == labels.numpy()).sum() / len(labels))

    # for dt in tqdm(m.estimators_):
    for dt in m.estimators_:
        best_feature = []
        best_acc = 0

        for _ in range(100):
            dt.select_features()
            for images, labels in trainloader:
                dt.fit(images.view(images.size(0),-1).numpy(), labels.numpy())
            for images, labels in testloader:
                pred = m.predict(images.view(images.size(0),-1).numpy())
                acc = (pred == labels.numpy()).sum() / len(labels)
                if acc > best_acc:
                    best_feature = dt.selected_features
                    best_acc = acc
        dt.selected_features = best_feature
        for images, labels in trainloader:
            dt.fit(images.view(images.size(0),-1).numpy(), labels.numpy())

    for images, labels in testloader:
        pred = m.predict(images.view(images.size(0),-1).numpy())
        acc = (pred == labels.numpy()).sum() / len(labels)
        print(f"compact acc: {acc}")

    ttt = m.estimators_
    sparsity = []
    for tt in ttt:
        sp, table = utils.find_sparsity(tt.DT)
        sparsity.append(sp)
    print(f"compact sparsity: {np.mean(sparsity)}")
