import os
import glob
import itertools

import numpy as np

DRY_RUN = True
VERBOSE = False

#SELECTED_MODE = "absolute"
#PROPORTIONS = [("test", 200), ("validation", 200)]

SELECTED_MODE = "percentage"
PROPORTIONS = [("test", 0.1), ("validation", 0.1)]

# define class names
class_names = [os.path.basename(path[:-1]) for path in glob.glob("train/*/")]
print("")
print("Detected classes:", len(class_names))
print("Classes:", ", ".join(class_names))
print("")

# get all files
all_files = glob.glob("train/*/*.*")
class_counts = {class_name: sum(1 for file_name in all_files if class_name in file_name) for class_name in class_names}
print("Sample distribution per class:")
for class_name in class_names:
    print(f"  - {class_name:17} : {class_counts[class_name]}")
print("")

# randomize files
np.random.shuffle(all_files)

# process all files
for current_class, (split_mode, split_proportion) in itertools.product(class_names, PROPORTIONS):
    # print report
    print(f"Processing: {split_mode} {current_class}  ...", end=" ")

    # count how many files to take
    take_count = 0
    if SELECTED_MODE == "absolute":
        take_count = split_proportion
    else:
        take_count = int(split_proportion * class_counts[current_class])

    # create a new folder for test files
    destination_folder = "{}/{}".format(split_mode, current_class)
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # move files
    processed_count = 0
    processed_paths = []
    for file_name in all_files:
        # check if we have taken enough data
        if processed_count >= take_count:
            break

        # check if the path contains the current class
        if current_class not in file_name:
            continue

        # get the new path
        destination_path = os.path.join(destination_folder, os.path.basename(file_name))

        if VERBOSE:
            print("Moving {} to {}".format(file_name, destination_path))

        if not DRY_RUN:
            os.rename(file_name, destination_path)

        processed_paths.append(file_name)
        processed_count += 1

    # print report
    print(processed_count, " OK!")

    # remove processed files
    all_files = [file_name for file_name in all_files if file_name not in processed_paths]

# print done
print("")
print("DONE!")
