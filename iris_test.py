#iris
import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree #decision tree classifier 
iris = load_iris()
test_idx = [0,50,100]

#train data
#delete example
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

#test data
#add out own example
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print (test_target)
print (clf.predict(test_data))

print (test_data[1] , test_target[1])
print (iris.feature_names, iris.target_names)