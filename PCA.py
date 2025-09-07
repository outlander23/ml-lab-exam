import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Parameters
img_size = (128, 128)
data_dir = "data/"  # folder with subfolders: "Cat", "Dog"

X = []
y = []

# Load images
for label, folder in enumerate(["Cat", "Dog"]):
    folder_path = os.path.join(data_dir, folder)
    for file in os.listdir(folder_path):
        try:
            img = imread(os.path.join(folder_path, file), as_gray=True)
            img_resized = resize(img, img_size)
            X.append(img_resized.flatten())  # flatten image
            y.append(label)
        except:
            continue

X = np.array(X)
y = np.array(y)

print("Original feature shape:", X.shape)  # (num_samples, 64*64)

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- PCA ---
pca = PCA(n_components=20)  # reduce to 50 components
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Train SVM on PCA features
clf_pca = SVC(kernel='rbf', C=10, gamma=0.01)
clf_pca.fit(X_train_pca, y_train)
y_pred_pca = clf_pca.predict(X_test_pca)
print("PCA + SVM Accuracy:", accuracy_score(y_test, y_pred_pca))

# --- LDA ---
lda = LDA(n_components=1)  # for 2 classes, max LDA component = 1
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

# Train SVM on LDA features
clf_lda = SVC(kernel='rbf', C=10, gamma=0.01)
clf_lda.fit(X_train_lda, y_train)
y_pred_lda = clf_lda.predict(X_test_lda)
print("LDA + SVM Accuracy:", accuracy_score(y_test, y_pred_lda))
