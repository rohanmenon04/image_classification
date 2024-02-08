import tensorflow as tf
from keras.optimizers import Adam
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten
from keras.metrics import Precision, Recall
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import time
import pathlib
import os

class TrainClassifier:
    """
    Trains an image classifier neural network
    """
    def __init__(self):
        self.model = Sequential([
        Input(shape=(150, 150, 3)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(6, activation='softmax')  # Assuming 6 classes
    ])
        self.model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy', Precision(), Recall()])
    def classify_training(self, path: str):
        """
        Retrieves a path to a directory of images and returns a train_generator image to be used for training
        :param path: path to image dataset for training
        :returns: train_generator object
        """
        datagen = ImageDataGenerator(rescale=1./255)
        train_generator = datagen.flow_from_directory(
            directory=path,
            target_size = (150, 150),
            batch_size = 32,
            class_mode = 'categorical')
        return train_generator
    def train(self, path_to_dataset: str, model_location: str):
        """
        Trains the specified neural network on an existing image dataset
        :param path_to_dataset: a string representing the path to the existing image dataset
        :param model_location: string representing the location the saved model should be written to, must have .keras at end
        """
        write_over = True
        if pathlib.Path(model_location).exists():
            print ('This model already exists')
            overwrite_perms = int(input('Input 1 if you would like to overwrite the existing dataset: '))
            if overwrite_perms != 1:
                write_over = False
        if write_over == True:
            train_generator = self.classify_training(path_to_dataset)
            start = time.time()
            self.model.fit(train_generator, epochs=10)
            end = time.time()
            self.model.save(model_location, overwrite=True)
            print (f'The model was trained in {(end-start)/60} minutes')
        else:
            print ('Program exiting...')

# train_classifier = TrainClassifier()
# model = train_classifier.train(path_to_dataset='image_dataset/seg_train/seg_train',
# model_location='src/models/initial_model.keras')