# CAM_DT
Random Forest for CAM.

This project is majorly using python.

## General Functions

1. Calculate the size of CAM needed for a given random forest.

2. Retrain a random forest so that it can have smaller sizes.

## Files

* compact_RF.py: the main file that takes in a random forest and offers a compact counterpart of it.
* modules.py: this file includes self defined modules: (1) Decision Tree and (2) Random Forest. Both modules are based on scikit-learn.
* utils.py: utility function needed.
* trained_RandomForest.joblib: a pretrained random forest with 100 trees each depth of 5.
* *.ipynb: debug files that can be ignored.