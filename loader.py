import os
from PIL import Image
import numpy as np
import random


def get_pictures(path):
    images = []
    imagepath = os.path.join(os.getcwd(), "data", path)
    for i, j, k in os.walk(imagepath):
        images = k
    return images


def image_vec(image):
    image = Image.open(image)
    vec_image = np.array(image)
    return vec_image


def label_vec(label, length):
    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    y = [np.zeros((1, len(alphabet)), dtype=np.uint8) for i in range(length)]
    for j, ch in enumerate(label):
        y[j][0, :] = 0
        y[j][0, alphabet.index(ch)] = 1
    return np.array(y)


def max_label_length(data):
    max_length = 1
    for i in data:
        label_length = len(i[:-4])
        if label_length > max_length:
            max_length = label_length
    return max_length


def load_dataset(data):
    images = get_pictures(data)
    length = max_label_length(images)
    data_len = len(images)
    data_set = []
    labels = []
    for image in images:
        image_path = os.path.join(os.getcwd(), "data", data, image)
        data_set.append(image_vec(image_path))
        # vec_label = label_vec(image[:-4], length)
        # labels.append(vec_label)
        label = image[:-4]
        labels.append(label)
    return data_len, length, data_set, labels


alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
HEIGHT = 40
WIDTH = 250
N_CLASS = 26
data_len, cata_len, X, Y = load_dataset('train')


def gen(batch_size=3):

    x = np.zeros((batch_size, HEIGHT, WIDTH, 3), dtype=np.uint8)
    y = [np.zeros((batch_size, N_CLASS), dtype=np.uint8) for i in range(cata_len)]
    while True:

        for i in range(batch_size):
            index = random.randint(0, data_len-1)
            x[i] = X[index]
            for j, ch in enumerate(Y[index]):
                y[j][i, :] = 0
                y[j][i, alphabet.index(ch)] = 1
        yield x, y

