import os
from .train_classifier import TrainClassifier
from keras.models import load_model, Model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
import pathlib
import numpy as np
from typing import List, Dict
from PIL import Image

class Classifier:
    """
    Loads a trained model and makes image classifications
    :param path: path to the pre-trained model
    :method predict_individual: makes a prediction of an individual jpg file
    :method find_accuracy: finds the accuracy of predictions across all classes
    :method find_specific_accuracy: finds how well the model is predicting each of the different classes
    :method check_dir: makes predictions for a directory full of random image files
    """
    def __init__(self, path: str):
        self.model: Model = load_model(path)
        self.map_function: Dict[int, str] = {0: 'buildings', 1: 'forest', 2: 'glacier', 3: 'mountain', 4: 'sea', 5: 'street'}
    def predict_individual(self, image_path: str) -> str:
        """
        Makes a prediction of the image type, receives the path to an image as input
        :param image_path: string containing the path to an image
        :returns: a string representing what the image is classified as
        """
        img: Image = image.load_img(path=image_path, target_size=(150, 150, 3))
        img_array: np.ndarray = image.img_to_array(img)
        img_array: np.ndarray = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        
        predictions: np.ndarray = self.model.predict(img_array, verbose=0)
        
        predicted_class: int = np.argmax(predictions, axis=1)
        
        for key, image_type in self.map_function.items():
            if predicted_class == key:
                return image_type.capitalize()

    def find_accuracy(self, dir_path: str) -> str:
        """
        This method computes how well the model predicts the different classes
        :param dir_path: this is the path to the directory containing the images to classify
        returns: a measure of the number of images classified correctly
        """
        correct: int = 0
        test_datagen: ImageDataGenerator = ImageDataGenerator(rescale=1./255)
        test_generator: DirectoryIterator = test_datagen.flow_from_directory(directory=dir_path, 
                                                          target_size=(150, 150), 
                                                          batch_size=32, 
                                                          class_mode='categorical',
                                                          shuffle=False)
        
        predictions: np.ndarray = self.model.predict(test_generator, verbose=0)
        predicted_class: int = np.argmax(predictions, axis=1)

        for index, prediction in enumerate(predicted_class):
            if prediction == test_generator.classes[index]:
                correct += 1
        
        return f'{(correct/len(predicted_class)*100):.2f}% classified correctly' if len(predicted_class) == len(test_generator.classes) else 'Mismatch of arrays'
    
    def find_specific_accuracy(self, dir_path: str, classification: str) -> None:
        """
        This method finds the accuracy of predicting a specific class, it receives the path to the folder and the category the image is looking for
        :param dir_path: this is a string representing the path to the directory with the images to be classified
        :classification: this is the target classification
        """
        if classification not in self.map_function.values():
            return f'{classification} not found'
        for idx, category in self.map_function.items():
            if category == classification:
                index = idx
                break
        directory: List[str] = os.listdir(dir_path)
        correct: int = 0
        count: int = 1
        length: int = len(directory)
        for img in directory:
            image_path: str = f'{dir_path}/{img}'
            if self.predict_individual(image_path) == classification.capitalize():
                correct += 1
            if count % 100 == 0:
                print (f'Progress: {count}/{length}')
            count += 1
        print (f'Accuracy in predicting {classification} is {(correct/length)*100:.2f}%')
    
    def check_dir(self, folder_path: str) -> None:
        """
        Receives a folder full of only jpg files and attempts to classify each one
        :param folder_path: a string containing the path to the folder containing the images
        """
        entries = os.listdir(folder_path)
        count = 1
        entry_length = len(entries)
        for img in entries:
            path = f'{folder_path}/{img}'
            print (f'{img}: {self.predict_individual(image_path=path)}')
            if count % 100 == 0:
                print (f'\n\n\nProgress: {count}/{entry_length}\n\n\n')
            count += 1
        
classifier = Classifier(path='src/models/initial_model.keras')
print (classifier.predict_individual(image_path='image_dataset/seg_train/seg_train/glacier/187.jpg'))
# print (classifier.find_accuracy(dir_path='image_dataset/seg_test/seg_test'))


#classifier.check_dir(folder_path='image_dataset/seg_pred/seg_pred')
#classifier.find_specific_accuracy(dir_path='image_dataset/seg_test/seg_test/glacier', classification='glacier')
#print (classifier.predict_individual('image_dataset/seg_pred/seg_pred/24227.jpg'))