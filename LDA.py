import numpy as np

def LDA(X, y, n_components):
    classes = np.unique(y)
    mean_overall = np.mean(X, axis=0)
    n_features = X.shape[1]

    # Scatter matrices
    Sw = np.zeros((n_features, n_features))
    Sb = np.zeros((n_features, n_features))

    for c in classes:
        X_c = X[y == c]
        mean_c = np.mean(X_c, axis=0)
        Sw += np.dot((X_c - mean_c).T, (X_c - mean_c))
        n_c = X_c.shape[0]
        mean_diff = (mean_c - mean_overall).reshape(-1, 1)
        Sb += n_c * np.dot(mean_diff, mean_diff.T)

    # Solve generalized eigenvalue problem
    eigvals, eigvecs = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))

    # Sort by eigenvalues
    sorted_idx = np.argsort(abs(eigvals))[::-1]
    eigvecs = eigvecs[:, sorted_idx]
    eigvecs = eigvecs[:, :n_components]

    return np.dot(X, eigvecs)

# Example
X = np.array([[4,2],[2,4],[2,3],[3,6],[4,4],[9,10],[6,8],[9,5],[8,7],[10,8]])
y = np.array([0,0,0,0,0,1,1,1,1,1])
print(LDA(X, y, 1))
