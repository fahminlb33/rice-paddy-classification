import os
import glob

import numpy as np


mode_percentages = [("test", 0.1), ("validation", 0.1)]
class_names = ["Bacterialblight", "Blast", "Brownspot", "Tungro", "Healthy"]

# count files for each class
# class_counts = {class_name: len(glob.glob("train/{}/*.jpg".format(class_name))) for class_name in class_names}
# print(class_counts)

for current_class in class_names:
    # get files for current class
    current_files = glob.glob("train/{}/*.*".format(current_class))
    total_original_files = len(current_files)
    
    # select take_test number of random element from current_files
    np.random.shuffle(current_files)

    for mode, percentage in mode_percentages:
        take_count = int(total_original_files * percentage)
        selected_files = current_files[:take_count]
        count = 0

        # create a new folder for test files
        destination_folder = "{}/{}".format(mode, current_class)
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        # move test files to test folder
        for test_file in selected_files:
            destination_path = os.path.join(destination_folder, os.path.basename(test_file))
            #print("Moving {} to {}".format(test_file, destination_path))

            os.rename(test_file, destination_path)
            count += 1

        # remove moved files
        current_files = [file for file in current_files if file not in selected_files]

        # print total moved
        print("Mode: {}".format(mode))
        print("Class: {}".format(current_class))
        print("Total moved: {}\n".format(count))
