import numpy as np

class DecisionTree:
    def __init__(self, X, y, max_depth=2,  min_leaf_size=1, depth=0, classes=None):
        self.X = np.array(X)
        self.y = np.array(y)
        self.N =  self.X.shape[0] # the number of samples
        self.depth = depth   # the depth of the node
        
        
        # Creating a list of possible classes.
        if classes is None: 
            self.classes = np.unique(self.y)
        else:
            self.classes = classes
            
            
        # Number of samples of each class present in the current node.
        self.class_count = []
        for i in self.classes:
            self.class_count.append(np.sum(i==self.y))
        self.class_count = np.array(self.class_count)    
        
        
        #Most frequent class in the current node.
        self.prediction = self.classes[np.argmax(self.class_count)]
        
        
        #Gini impurity for the current node.
        class_ratios = self.class_count / self.N
        self.gini = 1 - np.sum(class_ratios **2)
        
        
        #Part f)
        if (depth == max_depth) or ( self.gini == 0):
            self.axis = None
            self.t = None
            self.left = None
            self.right = None
            return
        
        
        best_gini = 2
        self.axis = 0
        self.t = 0
        
        
        for k in range(self.X.shape[1]):
            col_values = self.X[:, k].copy()
            col_values = np.sort(col_values)
                
            for j in range (len(col_values)):
                sel = self.X[:,k] <= col_values[j]
                n_left = np.sum(sel)
                n_right = np.sum(~sel)
                
                if (n_left >= min_leaf_size) & (n_right >= min_leaf_size):
                    _,left_counts = np.unique(self.y[sel], return_counts=True)
                    class_ratios = left_counts / n_left
                    left_gini = 1-np.sum((class_ratios **2))
                    
                    _,right_counts = np.unique(self.y[~sel], return_counts=True)
                    class_ratios = right_counts / n_right
                    right_gini = 1-np.sum((class_ratios **2))
                    gini = (n_left * left_gini + n_right * right_gini) / (n_left + n_right)
                    
                    if (gini <= best_gini):
                        best_gini = gini
                        self.axis = k
                        if((j +1)== len(col_values)):
                            self.t = col_values[j]
                        else:
                            self.t = (col_values[j]+col_values[j+1])/2
                            
                            
        sel = self.X[:,self.axis]<= self.t
        
        
        if (best_gini == 2) or (np.sum(sel) < min_leaf_size) or (np.sum(~sel) < min_leaf_size):
            self.axis = None
            self.t = None
            self.left   = None
            self.right = None
            return


        self.left = DecisionTree(self.X[sel,:], self.y[sel], max_depth, min_leaf_size,depth+1, self.classes)
        self.right = DecisionTree(self.X[~sel,:], self.y[~sel], max_depth, min_leaf_size,depth+1, self.classes)
         
                
    def classify_row(self,row):
        row = np.array(row)
        
        if self.left == None or self.right == None:
            return self.prediction 
    
        if row[self.axis] <= self.t:
            return self.left.classify_row(row)
        else:
            return self.right.classify_row(row)           
        
        
    def predict(self,X):
        self.X = np.array(X)
        predictions = []
        
        for i in range(0, len(X)):
            row = self.X[i,:]
            predictions.append(self.classify_row(row))
            
        predictions = np.array(predictions)
        return predictions
    
    
    def score(self,X,y):
        X = np.array(X)
        y = np.array(y)
        N = X.shape[0]
        y_pred = self.predict(X)     
        accuracy = (np.sum(y== y_pred))/ N   
        return accuracy
        
        
    def print_tree(self):
        msg = '  ' * self.depth + '* '
        msg += 'Size: ' + str(self.N) + ' '
        msg += str(list(self.class_count))
        msg += ', Gini: ' + str(round(self.gini,2))           
        if(self.left != None):
            msg += ', Axis: ' + str(self.axis)
            msg += ', Cut: ' + str(round(self.t,2))
        else:
            msg += ', Predicted Class: ' + str(self.prediction) #get the predicted class
                
        
        print(msg)
        
        if self.left != None:
            self.left.print_tree()
            self.right.print_tree()