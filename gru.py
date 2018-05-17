from keras.models import *
from keras.layers import *
from keras.callbacks import *
from captcha.image import ImageCaptcha
from random import choice
import random
from cnn import WIDTH, HEIGHT, N_LEN, characters, english_words


N_CLASS = 27+1


def gen_pic(batch_size=32):
    global conv_shape
    X = np.zeros((batch_size, WIDTH, HEIGHT, 3), dtype=np.uint8)
    y = np.zeros((batch_size, N_LEN), dtype=np.uint8)
    generator = ImageCaptcha(width=WIDTH, height=HEIGHT,
                             font_sizes=[25])

    while True:
        for i in range(batch_size):
            word = choice(english_words)
            word_len = len(word)
            X[i] = np.array(generator.generate_image(word)).transpose(1, 0, 2)
            for j, ch in enumerate(word):
                y[i][j] = characters.find(ch)

            if len(word) < N_LEN:
                for k in range(word_len, N_LEN):
                    y[i][k] = (characters.find("#"))

        yield [np.array(X), np.array(y), np.ones(batch_size)*int(conv_shape[1]-2),
               np.ones(batch_size)*N_LEN], np.ones(batch_size)


def init_model():

    rnn_size = 128

    input_tensor = Input((WIDTH, HEIGHT, 3))
    x = input_tensor
    for i in range(3):
        x = Convolution2D(64, (3, 3), activation='relu')(x)
        x = Convolution2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)

    conv_shape = x.get_shape()
    x = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2] * conv_shape[3])))(x)

    x = Dense(32, activation='relu')(x)

    gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(x)
    gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True,
                 kernel_initializer='he_normal', name='gru1_b')(x)
    gru1_merged = merge([gru_1, gru_1b], mode='sum')

    gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
    gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True,
                 kernel_initializer='he_normal', name='gru2_b')(gru1_merged)
    x = merge([gru_2, gru_2b], mode='concat')
    x = Dropout(0.25)(x)
    x = Dense(N_CLASS, kernel_initializer='he_normal', activation='softmax')(x)

    base_model = Model(inputs=input_tensor, outputs=x)

    labels = Input(name='the_labels', shape=[N_LEN], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([x, labels, input_length, label_length])

    model = Model(inputs=[input_tensor, labels, input_length, label_length], outputs=[loss_out])
    return conv_shape, base_model, model


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def evaluate(batch_num=1):
    global base_model

    batch_acc = 0
    generator = gen_pic()
    for i in range(batch_num):
        [X_test, y_test, _, _], _ = next(generator)
        y_pred = base_model.predict(X_test)
        print(y_test)
        shape = y_pred[:, 2:, :].shape
        out = K.get_value(K.ctc_decode(y_pred[:, 2:, :], input_length=np.ones(shape[0])*shape[1])[0][0])
        print('*'*20)
        print(out)
        # batch_acc += ((y_test == out).sum(axis=1) == 10).mean()
        batch_acc += np.array(y_test == out).mean()
    return batch_acc / batch_num


class Evaluate(Callback):
    def __init__(self):
        self.accs = []

    def on_epoch_end(self, epoch, logs=None):
        acc = evaluate() * 100
        self.accs.append(acc)
        print('acc: %f' % acc)


def train_model(model):
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adadelta')
    evaluator = Evaluate()
    earlystop = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
    model.fit_generator(gen_pic(), steps_per_epoch=3, epochs=2,
                        callbacks=[earlystop, evaluator],
                        validation_data=gen_pic(), validation_steps=128)
    model.save('ctc.h5')


def plot_model(model):

    from IPython.display import Image
    from keras.utils.vis_utils import plot_model
    plot_model(model, to_file="gru.png", show_shapes=True)
    Image('gru.png')


if __name__ == '__main__':
    global conv_shape, base_model, model
    conv_shape, base_model, model = init_model()
    train_model(model)
