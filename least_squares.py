import numpy as np
def least_squares(X, y):
    # Add bias term
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    # Normal Equation: (X^T X)^(-1) X^T y
    theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    return theta

# Example
X = np.array([[1],[2],[3],[4]])
y = np.array([2,3,4,5])  # y = x + 1
theta = least_squares(X,y)
print("Weights:", theta)
