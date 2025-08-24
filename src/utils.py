import os
import numpy as np

from skimage.color import rgb2gray, rgba2rgb
from skimage.io import imread
from skimage.transform import resize


def load_image(filepath, image_size=(64, 64)):
    image = imread(filepath)
    if len(image.shape) == 3:
        if image.shape[-1] == 4:
            image = rgba2rgb(image)
        image = rgb2gray(image)
    image_resized = resize(image, image_size, anti_aliasing=True)
    return image_resized.flatten()

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
                    image = load_image(filepath, image_size)
                    X.append(image)
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