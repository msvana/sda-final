import os
import pickle

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier

from utils import load_images, validate_dataset

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


if __name__ == "__main__":
    main()