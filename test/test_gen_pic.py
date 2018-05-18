import gen_pic
import unittest
import sys
sys.path.append('../')


class TestGen(unittest.TestCase):

    def test_gen(self):
        print('init....')
        path = '/Users/yinxingwei/Documents/deeplearning/BossSensor/data/'
        gen_pic.gen(path)
