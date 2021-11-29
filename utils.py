from sklearn import tree
from matplotlib import pyplot as plt
import numpy as np

def str2bool(a) -> bool:
    """
        a stupid function that parses "True"(str) to True(bool)
        because the argparse package is silly.
    """
    if a == "True":
        return True
    elif a == "False":
        return False
    else:
        raise NotImplementedError(f"{a}")

def map_CAM(DT: tree.DecisionTreeClassifier) -> np.ndarray:
    """
        maps a Decision Tree to CAM
    """
    flag = False
    tree_text = tree.export_text(DT)
    lines = tree_text.split("\n")
    depth = DT.get_depth()
    table = []
    visited = []
    dog = []
    catch = []
    for l in lines[:-1]:
        app = depth - (l.find("feature") - 1)//4
        if app == depth + 1:
            dog.append(-1)
            this_line = [0 for _ in range(DT.n_features_in_)]
            for index, this_depth in visited:
                this_line[index] = 1
            # print(visited)
            if len(visited) != 0:
                end = visited[-1][1]
            if flag:
                end = catch[-1][1]
            for i in range(end, depth):
                for j in catch[::-1]:
                    if j[1] == i:
                        this_line[j[0]] = 1
            table.append(this_line)
        else:
            if flag:
                flag = False
            left = l.find("_") + 1
            right = l.find("<=") if l.find("<=") != -1 else l.find(">")
            index = int(l[left:right])
            if len(visited) == 0:
                visited.append((index, app))
            else:
                if visited[-1] != (index,app):
                    visited.append((index, app))
                else:
                    poped = visited.pop()
                    catch.append(poped)
                    flag = True
    table = np.array(table)
    return (table > 0).astype(np.int32)

def find_sparsity(DT: tree.DecisionTreeClassifier):
    """
        Calculates the sparsity of a Decision Tree when mapped to CAM.
    """
    table = map_CAM(DT)
    empty = 0
    for i in range(len(table[0])):
        if sum(table[:,i]) == 0:
            empty += 1
    return 1 - ((table == 1).sum() / (table.size - empty * len(table))), table

def find_size(DT: tree.DecisionTreeClassifier):
    """
        Calculates the size of a Decision Tree when mapped to CAM.
    """
    return DT.get_n_leaves() * len(find_features(DT))

def find_features(DT: tree.DecisionTreeClassifier) -> list:
    """
        Find all features used in a decision tree.
    """
    tree_text = tree.export_text(DT)
    lines = tree_text.split("\n")
    depth = DT.get_depth()
    features = []
    for l in lines[:-1]:
        if l.find("class") == -1:
            app = depth - (l.find("feature") - 1)//4
            left = l.find("_") + 1
            right = l.find("<=") if l.find("<=") != -1 else l.find(">")
            index = int(l[left:right])
            features.append(index)
    return list(set(features))