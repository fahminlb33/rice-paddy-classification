import time
import random

import mlflow

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay

import tensorflow as tf
import mlflow.tensorflow

from training_params import BATCH_SIZE, RANDOM_SEED, IMG_SIZE, IMG_SHAPE, EPOCHS, LEARNING_RATE, clean_temp_dir

# mlflow tracking
RUN_NAME = "Finalized model"
EXPERIMENT_NAME = "baseline-amiril-no-augment"

# mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.tensorflow.autolog()

# Set seed
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

def preprocess_images(ds):
    ds = ds.shuffle(buffer_size=1024)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds

if __name__ == "__main__":
    clean_temp_dir()
    with mlflow.start_run(run_name=RUN_NAME):
        dataset_kwargs = {
            "label_mode": "categorical",
            "seed": RANDOM_SEED,
            "image_size": IMG_SIZE,
            "batch_size": BATCH_SIZE
        }
        mlflow.log_params(dataset_kwargs)

        train_dataset = tf.keras.utils.image_dataset_from_directory("dataset/train", **dataset_kwargs)
        test_dataset = tf.keras.utils.image_dataset_from_directory("dataset/test", **dataset_kwargs)
        validation_dataset = tf.keras.utils.image_dataset_from_directory("dataset/validation", **dataset_kwargs)
        
        # save class names
        class_names = train_dataset.class_names
        mlflow.log_params({"class_names": class_names})

        # create final model
        inputs = tf.keras.layers.Input(shape=IMG_SHAPE)
        x = tf.keras.layers.Conv2D(10, (3, 3), activation="relu")(inputs)
        x = tf.keras.layers.Conv2D(10, (3, 3), activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Dropout(0.01)(x)
        x = tf.keras.layers.Conv2D(20, (3, 3), activation="relu")(x)
        x = tf.keras.layers.Conv2D(20, (3, 3), activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Dropout(0.0001)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(300, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.01)(x)
        outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)
        model = tf.keras.Model(inputs, outputs)

        # save model structure
        tf.keras.utils.plot_model(model, "temp/arch.png", show_shapes=True)
        mlflow.log_artifact("temp/arch.png")

        joblib.dump(class_names, "temp/class_names.z")
        mlflow.log_artifact("temp/class_names.z")

        # define optimizer and loss function
        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        loss = tf.keras.losses.CategoricalCrossentropy()
        mlflow.log_params({
            "learning_rate": LEARNING_RATE,
            "optimizer": "SGD",
            "loss": "CategoricalCrossentropy"
        })

        # compile model
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        # save model summary
        summary_list = []
        model.summary(print_fn=lambda x: summary_list.append(x))
        mlflow.log_text("\n".join(summary_list), "model_summary.txt")

        # prepare dataset for training
        train_dataset = preprocess_images(train_dataset)
        validation_dataset = preprocess_images(validation_dataset)

        # run training
        time_start = time.time()
        H = model.fit(train_dataset, validation_data=validation_dataset, epochs=EPOCHS)
        time_end = time.time()
        mlflow.log_metric("training_time", time_end - time_start)

        # evaluate model
        y_pred = []  # store predicted labels
        y_true = []  # store true labels

        # iterate over the dataset
        test_dataset_preprocessed = preprocess_images(test_dataset)

        time_start = time.time()
        for image_batch, label_batch in test_dataset_preprocessed:
            # append true labels
            y_true.append(tf.argmax(label_batch, axis=1))

            # compute predictions
            preds = model.predict(image_batch)
            preds = tf.argmax(preds, axis=1)

            # append predicted labels
            y_pred.append(preds)
        time_end = time.time()
        mlflow.log_metric("inference_time", time_end - time_start)

        # convert the true and predicted labels into tensors
        y_true = tf.concat([item for item in y_true], axis=0).numpy()
        y_pred = tf.concat([item for item in y_pred], axis=0).numpy()

        # save confusion matrix
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=class_names, ax=ax)
        mlflow.log_figure(fig, "confusion_matrix.png")

        # save accuracy
        accuracy = accuracy_score(y_true, y_pred)
        mlflow.log_metric("accuracy", accuracy)

        # save macro F1 score
        macro_f1 = f1_score(y_true, y_pred, average="macro")
        mlflow.log_metric("macro_f1", macro_f1)

        # save classification report
        report = classification_report(y_true, y_pred, target_names=class_names, digits=6)
        mlflow.log_text(report, "classification_report.txt")

        # save training history
        df_history = pd.DataFrame(H.history)
        df_history.index.name = "epoch"
        mlflow.log_text(df_history.to_csv(), "training_history.csv")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3))
        ax1.plot(H.history["loss"], label="Train")
        ax1.plot(H.history["val_loss"], label="Validation")
        ax1.set_xlabel("Epoch #")
        ax1.set_ylabel("Loss")
        ax1.legend(loc="lower left")

        ax2.plot(H.history["accuracy"], label="Train")
        ax2.plot(H.history["val_accuracy"], label="Validation")
        ax2.set_xlabel("Epoch #")
        ax2.set_ylabel("Accuracy")
        ax2.legend(loc="lower left")

        fig.tight_layout()
        mlflow.log_figure(fig, "training_history.png")

        # save model
        model.save("temp/model.h5")
        mlflow.log_artifact("temp/model.h5")
