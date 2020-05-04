import numpy as np
import glob
import os
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

# Constants
DATA_SOURCE = "../data"
TRAIN_FOLDER = "/train"
TEST_FOLDER = "/test"
IMG_IDENTIFIER = "*.png"

def load_data():
    # Save data paths
    train_path = DATA_SOURCE + TRAIN_FOLDER
    test_path = DATA_SOURCE + TEST_FOLDER

    x_data = []
    y_data = []

    # Load train data
    for file_path in glob.glob(os.path.join(train_path, '**', IMG_IDENTIFIER), recursive=True):
        img_type = type_from_src(file_path)
        image = load_img(file_path, color_mode="grayscale")
        img_array = img_to_array(image)

        x_data.append(img_array)
        y_data.append(img_type)


    print(x_data)

def type_from_src(image_path):
    img_path_segment = image_path.split("/")
    type_index = len(img_path_segment) - 2

    return img_path_segment[type_index]