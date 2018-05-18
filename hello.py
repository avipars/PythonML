from sklearn import tree
features = [[140,1],[130,1],[150,1],[170,0]] #smooth = 1 
labels = [0,0,1,1] #0 apple 1 orange
clf = tree.DecisionTreeClassifier() #rules
clf = clf.fit(features, labels) #training
print(clf.predict([[160,0]])) 
#ep 1 ml apple orange
