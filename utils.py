import os
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from skimage.io import imread
from skimage.transform import resize


def get_set_of_imgs(img, replication_factor):
    ans = [img]
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        fill_mode='nearest')
    for i in range(replication_factor):
        ans.append(datagen.random_transform(img))
    return ans


def read_index(index_file, img_shape, classes, replication_factor):
    index_x, index_y = [], []
    with open(index_file) as fp:
        for line in fp:
            class_name, path = line.split(',')
            path = path.split('\n')[0]
            img = imread(path)
            imgs = get_set_of_imgs(img, replication_factor)
            imgs = [resize(img, img_shape, mode='constant').astype('float32') for img in imgs]
            index_x.extend(imgs)
            index_y.extend([classes[class_name]] * len(imgs))
    return np.array(index_x), index_y


def read_index_icon(icon_dir, img_shape, classes, replication_factor):
    index_x, index_y = [], []
    for class_name in classes:
        if (os.path.exists(os.path.join(icon_dir, class_name + ".png"))):
            path = os.path.join(icon_dir, class_name + ".png")
            img = imread(path)
            imgs = get_set_of_imgs(img, replication_factor)
            imgs = [resize(img, img_shape, mode='constant').astype('float32') for img in imgs]
            index_x.extend(imgs)
            index_y.extend([classes[class_name]] * len(imgs))
    return np.array(index_x), index_y




