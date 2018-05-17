from keras.models import *
from keras.layers import *
from utils import get_english_words
import string
from captcha.image import ImageCaptcha
from random import choice

english_words = get_english_words()
HEIGHT = 40
WIDTH = 250
N_LEN = 10
N_CLASS = 27
characters = string.ascii_lowercase+'#'


def gen_cnn(batch_size=5):

    x = np.zeros((batch_size, HEIGHT, WIDTH, 3), dtype=np.uint8)
    y = [np.zeros((batch_size, N_CLASS), dtype=np.uint8) for i in range(N_LEN)]
    generator = ImageCaptcha(width=WIDTH, height=HEIGHT,
                             font_sizes=[25])
    while True:
        for i in range(batch_size):
            word = choice(english_words)
            word_len = len(word)
            x[i] = np.array(generator.generate_image(word))
            for j, ch in enumerate(word):
                y[j][i, :] = 0
                y[j][i, characters.find(ch)] = 1

            if len(word) < N_LEN:
                for k in range(word_len, N_LEN):
                    y[k][i, :] = 0
                    y[k][i, characters.find("#")] = 1

        yield x, y


def cnn_model():

    input_tensor = Input((HEIGHT, WIDTH, 3))
    x = input_tensor
    for i in range(3):
        x = Convolution2D(64, (3, 3), activation='relu')(x)
        x = Convolution2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dropout(0.25)(x)
    x = [Dense(N_CLASS, activation='softmax', name='c%d' % (i + 1))(x) for i in range(N_LEN)]
    model = Model(inputs=input_tensor, outputs=x)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    return model


def decode(y):
    y = np.argmax(np.array(y), axis=2)[:, 0]
    return ''.join([characters[x] for x in y]).strip('#')


def plot_model():
    model = cnn_model()
    from IPython.display import Image
    from keras.utils.vis_utils import plot_model
    plot_model(model, to_file="model.png", show_shapes=True)
    Image('model.png')


def train():
    model = cnn_model()
    model.fit_generator(gen_cnn(), steps_per_epoch=50, epochs=3,
                        validation_data=gen_cnn(), validation_steps=12)
    model.save('model.h5')
    return model


def evaluate(model, batch_num=20):
    batch_acc = 0
    generator = gen_cnn(1)
    for i in range(batch_num):
        x, y = generator.__next__()
        y_pred = model.predict(x)
        print("y", decode(y))
        print('*' * 40)
        print("y_pred", decode(y_pred))
        batch_acc += np.mean(list(map(np.array_equal,
                                      np.argmax(y, axis=2).T,
                                      np.argmax(y_pred, axis=2).T)))
    return batch_acc / batch_num


if __name__ == '__main__':

    model = train()
    acc = evaluate(model)
    print('*'*40)
    print(acc)

