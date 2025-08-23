import os
import pickle

import numpy as np
from skimage.color import rgb2gray, rgba2rgb
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from skimage.io import imread
from skimage.transform import resize
from sklearn.ensemble import RandomForestClassifier


def main():
    X, y = load_images("images/")
    validate_dataset(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    pca = PCA(50)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    assert np.sum(pca.explained_variance_ratio_) > 0.8

    model = RandomForestClassifier(250)
    model.fit(X_train_pca, y_train)

    score_train = model.score(X_train_pca, y_train)
    score_test = model.score(X_test_pca, y_test)

    print(score_train, score_test)

    assert score_train > 0.90
    assert score_test > 0.40

    f = open("randomforest.pickle", "wb")
    model_pipeline = {"pca": pca, "model": model}
    pickle.dump(model_pipeline, f)
    f.close()


def load_images(directory, image_size=(64, 64), max_file_size_kb=500):
    X = []
    y = []

    for class_name in os.listdir(directory):
        print(class_name)
        class_dir = os.path.join(directory, class_name)
        if os.path.isdir(class_dir):
            for filename in os.listdir(class_dir):
                if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                    filepath = os.path.join(class_dir, filename)
                    file_size_kb = os.path.getsize(filepath) / 1024
                    if file_size_kb > max_file_size_kb:
                        continue
                    image = imread(filepath)
                    if len(image.shape) == 3:
                        if image.shape[-1] == 4:
                            image = rgba2rgb(image)
                        image = rgb2gray(image)
                    image_resized = resize(image, image_size, anti_aliasing=True)
                    X.append(image_resized.flatten())
                    y.append(class_name)

    return np.array(X), np.array(y)

def validate_dataset(X, y, image_size=(64, 64)):
    classes = set(y)
    assert len(classes) == 10

    for c in classes:
        X_c = X[y == c]
        assert len(X_c) >= 750
    
    for image in X[:100]:
        assert image.shape[0] == image_size[0] * image_size[1]


if __name__ == "__main__":
    main()