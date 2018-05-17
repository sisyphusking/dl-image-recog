import numpy as np
import unittest


length = 8
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
                'v', 'w', 'x', 'y', 'z']
y = [np.zeros((1, len(alphabet)), dtype=np.uint8) for i in range(length)]
# print(y)
label = 'about'
for j, ch in enumerate(label):
    y[j][0, :] = 0
    y[j][0, alphabet.index(ch)] = 1
    print(y[j])
    print(np.shape(y[j]))
    print(type(y[j]))

print(len(np.array(y)))
