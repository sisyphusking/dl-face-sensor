from gen_pic import gen
import unittest


class TestGen(unittest.TestCase):

    def test_gen(self):
        print('init....')
        path = '/Users/yinxingwei/Documents/deeplearning/BossSensor/data/'
        gen(path)
