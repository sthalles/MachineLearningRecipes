# Basic machine learning recipe using a decision tree classifier over the IRIS dataset

# 1- Import the dataset
from sklearn.datasets import load_iris
import numpy as np
from sklearn import tree
iris = load_iris()

# print iris.feature_names
# print iris.target_names
#
# print iris.data[0]
# print iris.target[0]

# print out the entire dataset
for i in range(len(iris.target)):
    print ("%d\t%s\t%s" % (i, iris.data[i], iris.target[i]))

# 2- Train a decision tree classifier

# create a separate testing data
test_ids = [0,50,100]

# training data
training_target = np.delete(iris.target, test_ids)
training_data = np.delete(iris.data, test_ids, axis=0)

# testing data
testing_target = iris.target[test_ids]
testing_data = iris.data[test_ids]

# 3- Predict label for new flowers

# Create a decision tree classifier
classifier = tree.DecisionTreeClassifier()

# train the classisfier using the training data
classifier.fit(training_data, training_target)

print testing_target
print classifier.predict(testing_data)

# 4- Visualize the tree
import pydot
from sklearn.externals.six import StringIO

dot_data = StringIO()
tree.export_graphviz(classifier, out_file=dot_data,
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True, rounded=True,
                         impurity=False)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")