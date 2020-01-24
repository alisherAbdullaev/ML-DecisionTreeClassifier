# This project provides an implementation of decision tree classifier. It supports an arbitary number of feature inputs, any number of classes, and accepts un-encoded class labels.

***

### DecisionTree.py

* The constructor ` __init__() ` takes parameter **self** that will be an instatnce of the DecisionTree class. The **X** and **y** parameters store the training data. The parameters **max_depth** and **min_leaf_size** are used to control the size of the tree, while **depth** records the depth of the current node. The parameter **classes** is used to pass an array of possible classes down from the root node to the lower nodes.

* Method `classify_row()` has two parameters **self** and **row**. The parameter **row** is an array of feature values for a single observation to be classified. This method is called from a leaf node and is to determine if row belongs to **self.right** or **self.right**.

* Method `predict()` returns an array with appended predictions based on classified rows from **X**.

* Method `score()` predicts labels using `predict()`. The predictions are used to calculate the model's accuracy on the supplied dataset. 

* Method `print_tree()` prints a summary of all the nodes in the tree as well as child nodes(if presenet). The information displays the number of observations, class distribution, and Gini score. For non-leaf nodes, the axis and location of the split is displayed whereas predicted class is displayed for leaf nodes.

***

### Testing.ipynb

* This jupyter notebook provides some examples with three different datasets. 

***

### ClassificationPlotter.py

This python file helps to visualize the data points.
