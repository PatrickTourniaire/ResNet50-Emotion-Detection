import glob
import os
import shutil

INDETIFIER = "*.png"

DATA_SOURCE = "../../data/gestures/original/leapgestrecog/leapGestRecog"
TRAIN_SOURCE = "../../data/gestures/train"
TEST_SOURCE = "../../data/gestures/test"

TOTAL_SECTIONS = 9
NUM_OF_TRAIN = 8

IMAGES_OF = ["palm", "l", "fist", "fist_moved", "thumb", "index", "ok", "palm_moved", "c", "down"]

# Creates folder sections in train and test
def create_folders():
    for image_type in IMAGES_OF:
        os.mkdir(TRAIN_SOURCE + "/" + image_type)
        os.mkdir(TEST_SOURCE + "/" + image_type)

# Finds what the image is of
def image_of(filename):
    return (int(filename.split("_")[2]) - 1)

# Places all train images
for i in range(0, NUM_OF_TRAIN):
    folder = DATA_SOURCE + "/0" + str(i)
    for file_path in glob.glob(os.path.join(folder, '**', INDETIFIER), recursive=True):
        filename = os.path.basename(file_path)
        type_index = image_of(filename)
        shutil.copy(file_path, TRAIN_SOURCE + "/" + IMAGES_OF[type_index])


folder = DATA_SOURCE + "/09"
for file_path in glob.glob(os.path.join(folder, '**', INDETIFIER), recursive=True):
    filename = os.path.basename(file_path)
    type_index = image_of(filename)

    shutil.copy(file_path, TEST_SOURCE + "/" + IMAGES_OF[type_index])