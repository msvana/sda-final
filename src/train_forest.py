import os
import numpy as np

from skimage.color import rgb2gray, rgba2rgb
from skimage.io import imread
from skimage.transform import resize

def main():
    X, y = load_images("images/")


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


if __name__ == "__main__":
    main()