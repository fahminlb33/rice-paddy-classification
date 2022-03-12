import os
import glob

class_names = ["Bacterialblight", "Blast", "Brownspot", "Tungro"]
counter = {class_name: 0 for class_name in class_names}

for mode in ["test", "validation"]:
    for current_class in class_names:
        # get files for current class
        current_files = glob.glob("{}/{}/*.jpg".format(mode, current_class))

        # crate train folder if not exists
        train_folder = "train/{}".format(current_class)
        if not os.path.exists(train_folder):
            os.makedirs(train_folder)

        # move test files to train folder
        for test_file in current_files:
            destination_path = os.path.join(train_folder, os.path.basename(test_file))
            #print("Moving {} to {}".format(test_file, destination_path))

            os.rename(test_file, destination_path)
            counter[current_class] += 1

    # print total moved    
    print("Mode: ", mode)
    print("Total moved: {}\n".format(sum(counter.values())))

    # right align dictionary output in console
    max_key_length = max([len(key) for key in counter.keys()])
    for key, value in counter.items():
        print("{}: {}".format(key.rjust(max_key_length), value))

