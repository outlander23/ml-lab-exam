import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize,rotate
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# Parameters
img_size = (16, 16)  # resize all images
data_dir = "data/"  # dataset folder with two subfolders: class0, class1

X = []
y = []

# Load images from two folders
for label, folder in enumerate(["Cat", "Dog"]):
    path = os.path.join(data_dir, folder)
    for file in os.listdir(path):

        img = imread(os.path.join(path, file), as_gray=True)
        img_resized = resize(img, img_size)

        

        for i in range(10):
            U, S, Vt = np.linalg.svd(img, full_matrices=False)
            features =S[:20]
            X.append(features)
            y.append(label)


       

X = np.array(X)
y = np.array(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
