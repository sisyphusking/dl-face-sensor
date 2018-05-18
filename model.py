from keras.models import *
from keras.layers import *
from keras.optimizers import SGD
from keras.models import load_model
from keras import backend as K
from loader import data_set, IMAGE_SIZE, resize_with_pad


def cnn_model(data, nb_classes=2):
    model = Sequential()
    # border_mode: 输入和输出尺寸一致，这一层特定设置
    model.add(Conv2D(32, (3, 3), border_mode='same', input_shape=data[0][0].shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    # 打印出模型概述信息
    model.summary()
    return model


def train(data, model, batch_size=32, nb_epoch=20):
    # lr:学习率， decay：每次更新后学习率衰减值 ，momentum：动量参数，nesterov：是否使用Nesterov动量
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # categorical_crossentropy: 多分类损失函数 ，accuracy：性能评估函数
    model.compile(loss='categorical_crossentropy',
                       optimizer=sgd,
                       metrics=['accuracy'])
    model.fit(data[0][0], data[0][1],
                       batch_size=batch_size,
                       nb_epoch=nb_epoch,
                       validation_data=(data[1][0], data[1][1]),
                       shuffle=True)
    model.save('model.h5')
    return model


def _predict(model, image):

    if K.image_dim_ordering() == 'th' and image.shape != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
        image = resize_with_pad(image)
        image = image.reshape((1, 3, IMAGE_SIZE, IMAGE_SIZE))
    elif K.image_dim_ordering() == 'tf' and image.shape != (1, IMAGE_SIZE, IMAGE_SIZE, 3):
        image = resize_with_pad(image)
        image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))
    image = image.astype('float32')
    image /= 255
    # 产生输入数据属于各个类别的概率，一般可以不需要这一步，除非想知道分类概率
    result = model.predict_proba(image)
    print(result)
    # 产生类别预测结果
    result = model.predict_classes(image)

    return result[0]


def evaluate(model, data):
    # model.evaluate返回一个测试误差的标量值，或者是一个list(其他评价指标)
    # verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
    score = model.evaluate(data[2][0], data[2][1], verbose=0)
    # model.metrics_names将给出list中各个值的含义，这里是accuracy
    print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))


def load():
     return load_model('model.h5')


def plot(model):
    from IPython.display import Image
    from keras.utils.vis_utils import plot_model
    plot_model(model, to_file="model.png", show_shapes=True)
    Image('model.png')
