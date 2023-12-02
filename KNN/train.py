# Import necessary libraries
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from KNN import KNN  

# Define a colormap for plotting
cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Load the Iris dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Plot the features using a scatter plot
plt.figure()
plt.scatter(X[:, 2], X[:, 3], c=y, cmap=cmap, edgecolors='k', s=20)
plt.show()

# Create an instance of the KNN classifier with k=5
clf = KNN(k=5)

# Train the KNN classifier on the training data
clf.fit(X_train, y_train)

# Make predictions on the test data
predictions = clf.predict(X_test)

# Print the predictions
print(predictions)
