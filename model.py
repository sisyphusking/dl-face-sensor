from keras.models import *
from keras.layers import *
from keras.optimizers import SGD
from keras.models import load_model
from keras import backend as K
from loader import data_set, IMAGE_SIZE, resize_with_pad


def cnn_model(data, nb_classes=2):
    model = Sequential()
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

    model.summary()
    return model


def train(data, model, batch_size=32, nb_epoch=20):
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
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
    result = model.predict_proba(image)
    print(result)
    result = model.predict_classes(image)

    return result[0]


def evaluate(model, data):
    score = model.evaluate(data[2][0], data[2][1], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))


def load():
     return load_model('model.h5')


if __name__ == '__main__':

    data = data_set('data/')
    model = cnn_model(data)
    model = train(data, model)
    evaluate(model, data)