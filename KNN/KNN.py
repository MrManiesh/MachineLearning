# Import necessary libraries
import numpy as np
from collections import Counter

# function to calculate the Euclidean distance between two points
def euclidean_distance(x1, x2):
    # Calculate the square root of the sum of squared differences between coordinates
    distance = np.sqrt(np.sum((x1 - x2) ** 2))
    return distance

# Implement the KNN (K-Nearest Neighbors) algorithm
class KNN:
    # Constructor with a default value of k set to 3
    def __init__(self, k=3):
        # Initialize the value of k
        self.k = k

    # Train the model with training data
    def fit(self, X, y):
        # Store the training data and corresponding labels
        self.X_train = X
        self.y_train = y

    # Make predictions on new data points
    def predict(self, X):
        # Generate predictions for each data point in the input array
        predictions = [self._predict(x) for x in X]
        return predictions
    
    # Helper method to predict the label for a single data point
    def _predict(self, x):
        # Compute the Euclidean distance between the input point and all training points
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # Get the indices of the k closest training points
        k_indices = np.argsort(distances)[:self.k]
        # Extract the labels of the k closest training points
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Find the most common label among the k nearest neighbors
        most_common = Counter(k_nearest_labels).most_common(1)[0][0]
        return most_common
