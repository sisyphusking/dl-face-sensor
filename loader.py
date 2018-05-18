import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import random
from keras import backend as K
from keras.utils import np_utils


WIDTH = 640
HEIGHT = 480

IMAGE_SIZE = 64

images = []
labels = []


def resize_with_pad(image, height=IMAGE_SIZE, width=IMAGE_SIZE):

    def get_padding_size(image):
        h, w, _ = image.shape
        longest_edge = max(h, w)
        top, bottom, left, right = (0, 0, 0, 0)
        if h < longest_edge:
            dh = longest_edge - h
            top = dh // 2
            bottom = dh - top
        elif w < longest_edge:
            dw = longest_edge - w
            left = dw // 2
            right = dw - left
        else:
            pass
        return top, bottom, left, right
    # top,bottom,left,right分别表示在原图四周扩充边缘的大小
    top, bottom, left, right = get_padding_size(image)
    BLACK = [0, 0, 0]
    # 扩充图像，保证高度和宽度等长
    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)
    # 图像缩放到height、width尺寸, 变成（64,64,3）
    resized_image = cv2.resize(constant, (height, width))

    return resized_image


def read_image(file_path):
    image = cv2.imread(file_path)
    image = resize_with_pad(image, IMAGE_SIZE, IMAGE_SIZE)

    return image


def load(path):
    for file_or_dir in os.listdir(path):
        abs_path = os.path.abspath(os.path.join(path, file_or_dir))
        if os.path.isdir(abs_path):  # dir
            load(abs_path)
        else:  # file
            if file_or_dir.endswith('.jpg'):
                image = read_image(abs_path)
                images.append(image)
                labels.append(path)

    return images, labels


# load函数读取图片的另外一种写法
# images = []
# labels = []
# import os
# def get_pictures(path):
#
#     imagepath = os.path.join(os.getcwd(), path)
#     for i, j, k in os.walk(imagepath):
#         for _ in k:
#             if _.endswith('.jpg'):
#                 images.append(os.path.join(i, _))
#                 labels.append(path)
#     return images, labels


def data_set(path, img_rows=IMAGE_SIZE, img_cols=IMAGE_SIZE, img_channels=3, nb_classes=2):

    images, labels = load(path)
    images = np.array(images)
    labels = np.array([0 for _ in labels])
    labels = np.reshape(labels, [-1])  # 变成n维数组

    # test_size: 测试集所占的比例， random_state：随机数种子（如果为定值，那么每次产生相同的随机数，如果不是，每次生成不同的随机数）
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3,
                                                        random_state=random.randint(0, 100))
    X_valid, X_test, y_valid, y_test = train_test_split(images, labels, test_size=0.5,
                                                        random_state=random.randint(0, 100))

    # 在如何表示一组彩色图片的问题上，theano和tensorflow有分歧，100张16×32（高*宽）的rgb图片，theano中表示的是（100,3,16,32）
    # 而在tensorflow中是（100,16,32,3），’th’模式是theano
    if K.image_dim_ordering() == 'th':
        X_train = X_train.reshape(X_train.shape[0], 3, img_rows, img_cols)
        X_valid = X_valid.reshape(X_valid.shape[0], 3, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 3, img_rows, img_cols)
        input_shape = (3, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
        X_valid = X_valid.reshape(X_valid.shape[0], img_rows, img_cols, 3)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)
        input_shape = (img_rows, img_cols, 3)

    # 将label变为向量
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_valid = np_utils.to_categorical(y_valid, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    X_train = X_train.astype('float32')  # 类型转换
    X_valid = X_valid.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255  # 归一化，将输入区间缩放到(0, 1)
    X_valid /= 255
    X_test /= 255

    return (X_train, Y_train), (X_valid, Y_valid), (X_test, Y_test)


if __name__ == '__main__':

    images, labels = data_set('data')[0]
    print(images[0])