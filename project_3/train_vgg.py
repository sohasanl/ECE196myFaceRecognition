"""
ECE196 Face Recognition Project
Author: W Chen

Use this as a template to:
1. load weights for vgg16
2. load images
3. finetune network
4. save weights
"""
import keras
import keras.utils
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.optimizers import SGD
from keras.layers import Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
import numpy as np
import glob
import os
import cv2
import random

IMG_H, IMG_W, NUM_CHANNELS = 224, 224, 3
MEAN_PIXEL = np.array([104., 117., 123.]).reshape((1, 1, 3))
path = os.getcwd()
print(path)
TRAIN_DIR = path + "/Data/Train"	#'../data/train'  # TODO
VAL_DIR = path + "/Data/Validation"		#'../data/validation'  # TODO
print(TRAIN_DIR)
NUM_EPOCHS = 5 # TODO
BATCH_SIZE = 16
NUM_CLASSES = 19  # TODO


def load_model():
    # TODO: use VGG16 to load lower layers of vgg16 network and declare it as base_model
    # TODO: use 'imagenet' for weights, include_top=False, (IMG_H, IMG_W, NUM_CHANNELS) for input_shape
    base_model = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(IMG_H, IMG_W, NUM_CHANNELS))
    print('Model weights loaded.')
    base_out = base_model.output
    # TODO: add a flatten layer, a dense layer with 256 units, a dropout layer with 0.5 rate,
    # TODO: and another dense layer for output. The final layer should have the same number of units as classes

    base_out = Flatten()(base_out)
    top_fc1 = Dense(256,activation = "relu")(base_out)
    top_fc1 = Dropout(0.5)(top_fc1)

    top_preds = Dense(NUM_CLASSES, activation="softmax")(top_fc1)
    model = Model(inputs=base_model.input, outputs=top_preds)

    model.summary()

    # TODO: compile the model, use SGD(lr=1e-4,momentum=0.9) for optimizer, 'categorical_crossentropy' for loss,
    # TODO: and ['accuracy'] for metrics


    sgd = SGD(lr=1e-4, momentum=0.9)
    model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])

    print ('Compile model')
    return model


def load_data(src_path):
    # under train/val/test dirs, each class is a folder with numerical numbers
    # X: number_images * height * width * channels
    # Y: number_images * 1
    class_path_list = sorted(glob.glob(os.path.join(src_path, '*')))
    image_path_list = []
    for class_path in class_path_list:
        image_path_list += sorted(glob.glob(os.path.join(class_path, '*jpg')))
    random.shuffle(image_path_list)
    num_images = len(image_path_list)
    X = np.zeros((num_images, IMG_H, IMG_W, NUM_CHANNELS))
    Y = np.zeros((num_images, 1))
    # read images and labels
    for i in range(num_images):
        image_path = image_path_list[i]
        label = int(image_path.split('/')[-2])
        image = cv2.imread(image_path, 1)
        image = cv2.resize(image, (IMG_H, IMG_W)) - MEAN_PIXEL
        X[i, :, :, :] = image
        Y[i, :] = label
    Y = to_categorical(Y, NUM_CLASSES)
    return X, Y


def main():
    # make model
    model = load_model()
    print ('VGG16 created\n')

    # read train and validation data and train the model for n epochs
    print ('Load train data:')
    X_train, Y_train = load_data(TRAIN_DIR)
    print ('Load val data:')
    X_val, Y_val = load_data(VAL_DIR)
    # TODO: Train model
    print(X_train.shape, Y_train.shape)

    #history = model.fit([X_train], [Y_train], nb_epoch=NUM_EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1)

    model.fit([X_train], [Y_train], batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_data=(X_val,Y_val))
    metrics = model.evaluate(X_train, Y_train)
    # TODO: Save model weights
    model.save_weights("model.h5")

    print ('model weights saved.')
    return


if __name__ == '__main__':
    main()
