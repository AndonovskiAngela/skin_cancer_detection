import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils import to_categorical

np.random.seed(21)


def build_cnn_model(input_shape=(224, 224, 3), num_classes=2):
    model = Sequential()

    # adding 64 filters, each filter has a size of 3*3
    # padding is of 2 types: SAME and VALID (SAME means doing the padding around the image, VALID means no padding)
    # kernel initilizer is for intializing the weights of the network --> the default one is glorot_uniform so don't need to mention parameter
    model.add(Conv2D(64, kernel_size=(3, 3), padding='Same', input_shape=input_shape, activation='relu',
                     kernel_initializer='glorot_uniform'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    # 25% of the nodes will be dropped out
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=(3, 3), padding='Same', activation='relu', kernel_initializer='glorot_uniform'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    # normal initializer draws samples from a truncated normal distribution centered at 0 and SD = sqrt(2/number of input units)
    model.add(Dense(128, activation='relu', kernel_initializer='normal'))
    model.add(Dense(128, activation='relu', kernel_initializer='normal'))

    model.add(Dense(num_classes, activation='softmax'))

    model.summary()

    ## OPTIMIZERS are the functions to adjust the weights and minimize the loss
    # Adam combines the best properties of the AdaGrad and RMSProp algorithms to provide an optimization algorithm that can handle sparse gradients on noisy problems.
    # Adam is relatively easy to configure where the default configuration parameters do well on most problems
    # lr is the alpha rate i.e. learning rate
    optimizer = Adam(lr=0.001)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=["accuracy"])

    return model


modelExists = False

try:
    model = load_model('./data/model/skin_cancer_model.keras')
    modelExists = True
except IOError:
    model = build_cnn_model()


def predict(image_array):
    return model.predict(image_array)


def train_model():
    if modelExists is False:
        directory_benign_train = './data/train/benign'
        directory_malignant_train = './data/train/malignant'
        directory_benign_test = './data/test/benign'
        directory_malignant_test = './data/test/malignant'

        ## Loading images and converting them to numpy array using their RGB value
        read = lambda imname: np.asarray(Image.open(imname).convert('RGB'))
        # np.asarray converts the objects into array/list

        # Loading train images
        img_benign_train = [read(os.path.join(directory_benign_train, filename)) for filename in
                            os.listdir(directory_benign_train)]
        img_malignant_train = [read(os.path.join(directory_malignant_train, filename)) for filename in
                               os.listdir(directory_malignant_train)]

        # Loading test images
        img_benign_test = [read(os.path.join(directory_benign_test, filename)) for filename in
                           os.listdir(directory_benign_test)]
        img_malignant_test = [read(os.path.join(directory_malignant_test, filename)) for filename in
                              os.listdir(directory_malignant_test)]

        # Converting list to numpy array for faster and more convenient operations going forward

        X_benign_train = np.array(img_benign_train, dtype='uint8')
        X_malignant_train = np.array(img_malignant_train, dtype='uint8')

        X_benign_test = np.array(img_benign_test, dtype='uint8')
        X_malignant_test = np.array(img_malignant_test, dtype='uint8')

        ## Creating labels: benign is 0 and malignant is 1

        y_benign_train = np.zeros(X_benign_train.shape[0])
        y_malignant_train = np.ones(X_malignant_train.shape[0])

        y_benign_test = np.zeros(X_benign_test.shape[0])
        y_malignant_test = np.ones(X_malignant_test.shape[0])

        ## Merge data to form complete training and test sets
        # axis = 0 means rows

        X_train = np.concatenate((X_benign_train, X_malignant_train), axis=0)
        y_train = np.concatenate((y_benign_train, y_malignant_train), axis=0)

        X_test = np.concatenate((X_benign_test, X_malignant_test), axis=0)
        y_test = np.concatenate((y_benign_test, y_malignant_test), axis=0)

        s1 = np.arange(X_train.shape[0])
        np.random.shuffle(s1)
        X_train = X_train[s1]
        y_train = y_train[s1]

        s2 = np.arange(X_test.shape[0])
        np.random.shuffle(s2)
        X_test = X_test[s2]
        y_test = y_test[s2]

        fig = plt.figure(figsize=(12, 8))
        columns = 5
        rows = 3

        for i in range(1, columns * rows + 1):
            ax = fig.add_subplot(rows, columns, i)
            if y_train[i] == 0:
                ax.title.set_text('Benign')
            else:
                ax.title.set_text('Malignant')
            plt.imshow(X_train[i], interpolation='nearest')

        y_train = to_categorical(y_train, num_classes=2)
        y_test = to_categorical(y_test, num_classes=2)

        X_train = X_train / 255
        X_test = X_test / 255

        # Learning rate annealer is used to reduce the learning rate by some percentage after certain number of training iterations/epochs
        learning_rate_annealer = ReduceLROnPlateau(monitor='val_acc',
                                                   patience=5,
                                                   verbose=1,
                                                   factor=0.5,
                                                   min_lr=1e-7)

        # epochs is the number of iterations
        # batch_size is the number of images in one epoch
        # verbose = 1 shows us the animation of the epoch using progres_bar
        model.fit(X_train,
                  y_train,
                  validation_split=0.2,
                  epochs=10,
                  batch_size=64,
                  verbose=1,
                  callbacks=[learning_rate_annealer])

        model.evaluate(X_test, y_test, 1000)

        model.save('./data/model/skin_cancer_model.keras')
