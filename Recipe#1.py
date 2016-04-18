# create a basic decision tree classifier

from sklearn import tree

# Step 1: collect training data (Example of the problem we want to solve)

# Tip #1: The more data we have, the better the classifier we can create
# The classifier learns by looking for pattern in the training data

# Feature vector [Weight (grames), Texture (0=Bumpy, 1=Smooth)]
# input of the classifier
# Use numbers 0 and 1 instead of lables (Bumpy or Smooth) because sklearn uses real-valued features

feature_names = ["Smooth", "Bumpy"]
target_names = ["Apple", "Orange"]

train_features = [[140, 1],
                  [130, 1],
                  [150, 0],
                  [170, 0]]

# training data 0-Apple, 1-Orange
train_labels =[0, 0, 1, 1]


test_data = [[160, 0],
             [165, 1],
             [200, 1],
             [140, 0]]

# train a classifier
# Decision tree classifier

# create an ampty decision tree classifier
classifier = tree.DecisionTreeClassifier()

# train the classifier by looking at the training data and finding patterns in it
# (In sklearn, the training algorithm is included in the classifier object)
classifier = classifier.fit(train_features, train_labels)

# Make predictions
for res in classifier.predict(test_data):
    print "Apple" if (res == 0) else "Orange"

# import pydot
# from sklearn.externals.six import StringIO
#
# dot_data = StringIO()
# tree.export_graphviz(classifier, out_file=dot_data,
#                          feature_names=feature_names,
#                          class_names=target_names,
#                          filled=True, rounded=True,
#                          special_characters=True)
# graph = pydot.graph_from_dot_data(dot_data.getvalue())
# graph.write_pdf("iris.pdf")