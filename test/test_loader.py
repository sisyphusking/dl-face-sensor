from loader import data_set
import unittest


class TestLoader(unittest.TestCase):

    def test_loader(self):
        path = '/Users/yinxingwei/Documents/deeplearning/BossSensor/data/'
        images, labels = data_set(path)[0]
        print(images[0])