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