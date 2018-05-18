from model import *
import unittest


class TestModel(unittest.TestCase):

    def test_model(self):
        path = '/Users/yinxingwei/Documents/deeplearning/BossSensor/data/'
        data = data_set(path)
        model = cnn_model(data)
        model = train(data, model)
        evaluate(model, data)