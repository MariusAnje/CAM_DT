import torch
import torchvision.transforms as transforms
import torchvision
import sklearn
from sklearn import tree
from sklearn import ensemble
from matplotlib import pyplot as plt
import utils
from utils import str2bool
import numpy as np
from joblib import dump, load
from tqdm import tqdm
import argparse
from modules import HanpiDecisionTree, MoRandomForest

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--iter', action='store', type=int, default=100,
            help='# of iterations for finding a DT')
    parser.add_argument('--n_features', action='store', type=int, default=5,
            help='# of features used in a DT')
    parser.add_argument('--pretrained', action='store',type=str2bool, default=True,
            help='if we use pretrained models')
    parser.add_argument('--shuffle', action='store',type=str2bool, default=False,
            help='if we shuffle the order of DTs in RF')
    
    args = parser.parse_args()
    print(args)

    addr = './LOG/211201_100iter.txt'

    # Loading MNIST dataset. Not that all data is made into a huge vector
    BS = 50000
    trainset = torchvision.datasets.MNIST(root='~/Private/data', train=True,
                                            download=False, transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BS,
                                            shuffle=True, num_workers=4)

    testset = torchvision.datasets.MNIST(root='~/Private/data', train=False,
                                        download=False, transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=BS,
                                                shuffle=False, num_workers=4)

    # Loading the original random forest
    if args.pretrained:
        # Loading the original random forest if "args.pretrained" is True
        clf = load('trained_RandomForest.joblib') 
    else:
        # Training a new random forest if "args.pretrained" is False
        clf = ensemble.RandomForestClassifier(max_depth=5)
        for images, labels in trainloader:
            clf.fit(images.view(images.size(0),-1).numpy(), labels.numpy())
        dump(clf, 'trained_RandomForest_new.joblib') # WARNING: we may need a new name here in order not to destroy the old one
        for images, labels in testloader:
            print(clf.score(images.view(images.size(0),-1).numpy(), labels.numpy()))

    # calculating the sparsity and sizes of each decision tree in the original random forest
    ttt = clf.estimators_
    sparsity = []
    table_size = []
    for tt in ttt: # each decision tree is calculated individually
        t_size = utils.find_size(tt)
        sp, table = utils.find_sparsity(tt)
        sparsity.append(sp)
        table_size.append(t_size)
    print(f"sparsity before compression: {np.mean(sparsity):.5f}") # the average size is shown
    print(f"CAM size before compression: {np.mean(table_size)}")

    
    # change all the decision trees to HanpiDecisionTree
    m = MoRandomForest(clf.estimators_)
    # calculate the accuracy of the original random forest
    for images, labels in testloader:
        pred = m.predict(images.view(images.size(0),-1).numpy())
        acc = (pred == labels.numpy()).sum() / len(labels)
        print(f"original acc: {acc}")

    ##LOG THIS
    with open(addr, 'a+') as filehandle:
            filehandle.write(f"sparsity before compression: {np.mean(sparsity):.5f}"+'\n')
            filehandle.write(f"CAM size before compression: {np.mean(table_size)}"+'\n')
            filehandle.write(f"original acc: {acc}"+'\n')

    # generate the order for which decision tree is pruned first
    ranking = list(range(len(m.estimators_)))
    if args.shuffle: # if "args.shuffle" is False, the training is done sequentially
        np.random.shuffle(ranking)

    ###Logging

    for index in ranking:  # for debuging, use tqdm
    # for index in ranking: # for each decision tree
        dt = m.estimators_[index]
        best_feature = []
        best_acc = 0

        records = []
        
        # for each decision tree, randomly find some features, train a new tree
        # find the best trained tree
        for _ in range(args.iter):
            dt.select_features(args.n_features)
            for images, labels in trainloader:
                dt.fit(images.view(images.size(0),-1).numpy(), labels.numpy())
            for images, labels in testloader:
                pred = m.predict(images.view(images.size(0),-1).numpy()) # the best trained tree in terms of the overall accuracy
                acc = (pred == labels.numpy()).sum() / len(labels)
                if acc > best_acc:
                    best_feature = dt.selected_features
                    best_acc = acc
            #Record
            records.append((dt.selected_features, acc))

        dt.selected_features = best_feature

        ##write in file
        with open(addr, 'a+') as filehandle:
            filehandle.write('___INDEX___'+'%d'%index+'___BESTACC___'+'%.5f'%best_acc+'___BESTFeature___'+'%s'%best_feature+'\n')
            for item in records:
                filehandle.write(f"{item}\n")
                filehandle.write('\n')

        for images, labels in trainloader:
            dt.fit(images.view(images.size(0),-1).numpy(), labels.numpy())

    # calculate the accuracy of the pruned random forest
    for images, labels in testloader:
        pred = m.predict(images.view(images.size(0),-1).numpy())
        acc = (pred == labels.numpy()).sum() / len(labels)
        print(f"compact acc: {acc}")

    # calculating the sparsity and sizes of each decision tree in the original random forest
    ttt = m.estimators_
    sparsity = []
    table_size = []
    for tt in ttt:
        t_size = utils.find_size(tt.DT)
        sp, table = utils.find_sparsity(tt.DT)
        sparsity.append(sp)
        table_size.append(t_size)
    print(f"sparsity after compression: {np.mean(sparsity):.5f}")
    print(f"CAM size after compression: {np.mean(table_size)}")

    ##LOG THIS
    with open(addr, 'a+') as filehandle:
            filehandle.write(f"sparsity after compression: {np.mean(sparsity):.5f}"{np.mean(sparsity):.5f}"+'\n')
            filehandle.write(f"CAM size after compression: {np.mean(table_size)}"+'\n')
            filehandle.write(f"compact acc: {acc}")
        
    dump(ttt, '1201_pruned_RandomForest.joblib')