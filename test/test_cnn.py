import unittest
from cnn import *


class TestCnn(unittest.TestCase):

    def test_cnn(self):
        model = cnn_model()
        model.fit_generator(gen(), steps_per_epoch=10, epochs=3,
                            validation_data=gen(), validation_steps=12)