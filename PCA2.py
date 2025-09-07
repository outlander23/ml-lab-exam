import numpy as np

def PCA(X, n_components):
    # Step 1: Standardize data
    X_meaned = X - np.mean(X, axis=0)

    # Step 2: Covariance matrix
    cov_matrix = np.cov(X_meaned, rowvar=False)

    # Step 3: Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Step 4: Sort eigenvectors by eigenvalues (descending)
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_idx]
    
    # Step 5: Select top components
    eigenvectors = eigenvectors[:, :n_components]

    # Step 6: Project data
    X_reduced = np.dot(X_meaned, eigenvectors)
    return X_reduced

# Example
X = np.random.rand(10, 5)  # 10 samples, 5 features
print(PCA(X, 2))
