#!/usr/bin/env python
# coding=utf-8

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.utils import np_utils
import string

chars = string.ascii_uppercase
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))


def load_data(look_back=3):

    data, label = [], []
    for i in range(0, len(chars)-look_back, 1):
        seq_in = chars[i:i+look_back]
        seq_out = chars[i+look_back]
        data.append([char_to_int[c] for c in seq_in])
        label.append(char_to_int[seq_out])
    return data, label


seq_length = 3
data, label = load_data(seq_length)
# seq_length这里等于look_back
X = np.reshape(data, (len(data), seq_length, 1))
X = X / float(len(chars))
y = np_utils.to_categorical(label) # shape:(23,26)

model = Sequential()
model.add(LSTM(32, input_shape=(X.shape[1], X.shape[2]))) # 23*3*1， input_shape:(3,1)
model.add(Dense(128, activation='relu'))
model.add(Dense(y.shape[1], activation='softmax'))  # 26维的列向量

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, nb_epoch=300, batch_size=32, verbose=2)
scores = model.evaluate(X, y, verbose=1)
print('accuracy: %.2f%%' % (scores[1]*100))

for p in data:
    x = np.reshape(p, (1, len(p), 1))  # p的shape:(3,1)-->(1,3,1)
    x = x /float(len(chars))
    y_pred = model.predict(x, verbose=0)
    index = np.argmax(y_pred) # 返回最大值的索引
    result = int_to_char[index]
    seq_in = [int_to_char[v] for v in p]
    print( seq_in, '-> ', result)

print(scores)