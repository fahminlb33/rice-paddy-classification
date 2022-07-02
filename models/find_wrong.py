import joblib

import tensorflow as tf

dataset_kwargs = {
    "label_mode": "categorical",
    "seed": 0,
    "image_size": (160, 160),
    "batch_size": 5,
    "shuffle": False
}

model = tf.keras.models.load_model("models/model.h5")
class_names = joblib.load("models/class_names.z")
ds = tf.keras.utils.image_dataset_from_directory("dataset/validation", **dataset_kwargs)
file_paths = ds.file_paths

ds = ds.map(lambda x, y: (tf.keras.applications.mobilenet_v2.preprocess_input(x), y), num_parallel_calls=None)

y_pred = []  # store predicted labels
y_true = []  # store true labels

for image_batch, label_batch in ds:
    # append true labels
    y_true.append(tf.argmax(label_batch, axis=1))

    # compute predictions
    preds = tf.nn.softmax(model.predict(image_batch))
    preds = tf.argmax(preds, axis=1)

    # append predicted labels
    y_pred.append(preds)

# convert the true and predicted labels into tensors
y_true = tf.concat([item for item in y_true], axis=0).numpy()
y_pred = tf.concat([item for item in y_pred], axis=0).numpy()

print("True labels:", y_true)
print("Predicted labels:", y_pred)

# find mismatch index in two list
mismatch_index = [i for i in range(len(y_true)) if y_true[i] != y_pred[i]]
print("Mismatch index:", mismatch_index)

# get filenames from tensorflow dataset based on mismatch index
filenames = [file_paths[i] for i in mismatch_index]
print("Filenames:", filenames)
