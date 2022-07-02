import os

BATCH_SIZE = 10
RANDOM_SEED = 0
IMG_SIZE = (160, 160)
IMG_SHAPE = (160, 160, 3)
EPOCHS = 50
LEARNING_RATE = 0.001

def delete_success(path):
    try:
        os.remove(path)
    except OSError:
        pass

def clean_temp_dir():
    delete_success("temp/model.h5")
    delete_success("temp/arch.png")
    delete_success("temp/class_names.z")
