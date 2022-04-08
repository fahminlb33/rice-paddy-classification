import os
from typing import Tuple, List

import joblib
import numpy as np
import matplotlib.pyplot as plt

from skimage.transform import resize

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import load_img, array_to_img, img_to_array

class PredictorModel:
    tf_model: Model = None
    class_names: List[str] = None
    initialized: bool = False
    IMG_SIZE: Tuple[int, int] = (160, 160)

    def load_model(self, model_root: str, tf_name: str, class_name: str) -> None:
        model_path = os.path.abspath(os.path.join(model_root, tf_name))
        name_path = os.path.abspath(os.path.join(model_root, class_name))

        self.tf_model = load_model(model_path)
        self.class_names = joblib.load(name_path)
        self.initialized = True

    def gradcam_heatmap(self, img_array, last_conv_layer_name="Conv_1", pred_index=None):
        convLayer = self.tf_model.get_layer(last_conv_layer_name).output
        grad_model = tf.keras.models.Model(
            [self.tf_model.inputs], [convLayer, self.tf_model.output]
        )

        with tf.GradientTape() as tape:
            last_conv_layer_logits, prediction_logits = grad_model(img_array)
            pred_index = tf.argmax(prediction_logits[0])
            loss_value = prediction_logits[:, pred_index]

            grads = tape.gradient(loss_value, last_conv_layer_logits)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

            heatmap = tf.matmul(last_conv_layer_logits[0], pooled_grads[..., tf.newaxis])
            heatmap = tf.squeeze(heatmap)
            heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)

            return heatmap.numpy()

    def create_heatmap(self, heatmap, image_size):
        # Use RGB values of the colormap
        nipy_cm = plt.cm.get_cmap("nipy_spectral")
        nipy_colors = nipy_cm(np.arange(256))[:, :3]
        heatmap_cm = nipy_colors[heatmap] * 255

        # Create an image with RGB colorized heatmap
        return resize(heatmap_cm, image_size, anti_aliasing=True)

    def create_superimposed_heatmap(self, original_img, heatmap, alpha=0.4):
        return heatmap * alpha + original_img

    def create_superimposed_mask(self, original_img, heatmap):
        # Median/Mean thresholding
        threshold_level = np.median(heatmap)
        heatmap_mask = heatmap[:, :, 0] < threshold_level

        heatmap_mask = heatmap_mask.astype(int)
        heatmap_mask[heatmap_mask == 1] = 255.0

        base_img = original_img.astype(int)
        return np.bitwise_or(heatmap_mask[:, :, np.newaxis], base_img)

    def internal_predict(self, image_path: str):
        # load image
        image_tensor = load_img(image_path, target_size=self.IMG_SIZE)
        image_tensor = img_to_array(image_tensor)
        image_tensor = tf.keras.applications.mobilenet.preprocess_input(image_tensor)
        image_tensor = tf.expand_dims(image_tensor, axis=0)

        # predict
        prediction = self.tf_model.predict(image_tensor)
        prediction = tf.nn.softmax(prediction)
        prediction = tf.argmax(prediction, axis=1)

        # run Grad-CAM algorithm
        gradcam = self.gradcam_heatmap(image_tensor)
        gradcam = np.uint8(255 * gradcam)

        return (prediction[0], gradcam)

    def predict(self, image_path: str, output_path: str) -> Tuple[str, str]:
        if not self.initialized:
            raise Exception("Predictor model not initialized")

        # Make prediction and get Grad-CAM
        (predicted, heatmap_raw) = self.internal_predict(image_path)

        # load original image
        original_img = load_img(image_path)
        original_img = img_to_array(original_img)
        image_size = (original_img.shape[0], original_img.shape[1])

        # create base heatmap using colormap
        heatmap_path = os.path.join(output_path, "heatmap.jpg")
        heatmap_arr = self.create_heatmap(heatmap_raw, image_size)
        heatmap_img = array_to_img(heatmap_arr)
        heatmap_img.save(heatmap_path)

        # superimpose the heatmap on original image
        heatmap_imposed_path = os.path.join(output_path, "superimposed.jpg")
        heatmap_imposed_arr = self.create_superimposed_heatmap(original_img, heatmap_arr, alpha=1.0)
        heatmap_imposed_img = array_to_img(heatmap_imposed_arr)
        heatmap_imposed_img.save(heatmap_imposed_path)

        # Superimpose the heatmap on original image
        masked_path = os.path.join(output_path, "masked.jpg")
        masked_arr = self.create_superimposed_mask(original_img, heatmap_arr)
        masked_img = array_to_img(masked_arr)
        masked_img.save(masked_path)

        return (
            self.class_names[predicted],
            heatmap_path,
            heatmap_imposed_path,
            masked_path
        )
