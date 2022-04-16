import logging
import os
from typing import Tuple, List

import joblib
import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread, imsave
from skimage.transform import resize

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import load_img, array_to_img, img_to_array

class PredictorService:
    MAX_HEIGHT: int = 800
    tf_model: Model = None
    class_names: List[str] = None
    initialized: bool = False
    IMG_SIZE: Tuple[int, int] = (160, 160)
    logger: logging.Logger

    def __init__(self) -> None:
        self.logger = logging.getLogger("PredictorService")

    def load_model(self, model_root: str, tf_name: str, class_name: str) -> None:
        # create model path using absolute path
        model_path = os.path.abspath(os.path.join(model_root, tf_name))
        name_path = os.path.abspath(os.path.join(model_root, class_name))

        # load the model
        self.tf_model = load_model(model_path)
        self.class_names = joblib.load(name_path)
        self.initialized = True

    def constrain_image_size(self, img_path: str) -> str:
        # load image
        img = imread(img_path)

        # check if image is too large
        if img.shape[0] <= self.MAX_HEIGHT:
            return img_path

        # calculate aspect ratio and new image dimensions
        aspect_ratio = img.shape[1] / img.shape[0]
        dim = (self.MAX_HEIGHT, int(aspect_ratio * self.MAX_HEIGHT))

        # resize image
        img = resize(img, dim)

        # save image
        new_name = os.path.splitext(img_path)[0] + "_resized.jpg"
        new_path = os.path.join(os.path.dirname(img_path), new_name)        
        imsave(new_path, img)

        return new_path

    def create_gradcam_matrix(self, img_array: np.ndarray, last_conv_layer_name="Conv_1") -> np.ndarray:
        # create model to output the last convolutional layer
        convLayer = self.tf_model.get_layer(last_conv_layer_name).output
        grad_model = tf.keras.models.Model(
            [self.tf_model.inputs], [convLayer, self.tf_model.output]
        )

        # get the gradients of the last convolutional layer
        with tf.GradientTape() as tape:
            # forward pass
            last_conv_layer_logits, prediction_logits = grad_model(img_array)
            pred_index = tf.argmax(prediction_logits[0])
            loss_value = prediction_logits[:, pred_index]

            # get the gradients of the last convolutional layer
            grads = tape.gradient(loss_value, last_conv_layer_logits)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

            # calculate Grad-CAM
            heatmap = tf.matmul(last_conv_layer_logits[0], pooled_grads[..., tf.newaxis])
            heatmap = tf.squeeze(heatmap)
            heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)

            return heatmap.numpy()

    def create_heatmap_from_gradcam(self, heatmap: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
        # use RGB values of the colormap
        nipy_cm = plt.cm.get_cmap("nipy_spectral")
        nipy_colors = nipy_cm(np.arange(256))[:, :3]
        heatmap_cm = nipy_colors[heatmap] * 255

        # create an image with RGB colorized heatmap
        resized = resize(heatmap_cm, image_size, anti_aliasing=True)
        resized = resized.astype(np.uint8)

        return resized

    def superimpose(self, original_img: np.ndarray, heatmap: np.ndarray, alpha=0.4) -> np.ndarray:
        # apply heatmap on top of original image
        return heatmap * alpha + original_img

    def mask_superimpose(self, original_img: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
        # median thresholding
        threshold_level = np.median(heatmap)
        heatmap_mask = heatmap[:, :, 0] < threshold_level

        # convert all pixels to black except for the heatmap
        heatmap_mask = heatmap_mask.astype(int)
        heatmap_mask[heatmap_mask == 1] = 255.0

        # use bitwise or to mask original image
        base_img = original_img.astype(int)

        # apply mask
        masked = np.bitwise_or(heatmap_mask[:, :, np.newaxis], base_img)
        masked = masked.astype(np.uint8)

        return masked

    def classify_and_gradcam(self, image_path: str) -> Tuple[int, np.ndarray]:
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
        gradcam = self.create_gradcam_matrix(image_tensor)
        gradcam = np.uint8(255 * gradcam)

        return (prediction[0], gradcam)

    def predict(self, image_path: str, output_path: str) -> Tuple[str, str, str, str]:
        if not self.initialized:
            raise Exception("Predictor model not initialized")

        # make prediction and get Grad-CAM
        self.logger.info(f"Running prediction using TensorFlow...")
        (predicted, gradcam) = self.classify_and_gradcam(image_path)

        # load original image
        self.logger.info(f"Loading original image...")
        original_img = load_img(image_path)
        original_img = img_to_array(original_img)
        image_size = (original_img.shape[0], original_img.shape[1])

        # create base heatmap using colormap
        self.logger.info(f"Creating heatmap from Grad-CAM...")
        heatmap_path = os.path.join(output_path, "heatmap.jpg")
        heatmap_arr = self.create_heatmap_from_gradcam(gradcam, image_size)
        heatmap_img = array_to_img(heatmap_arr)
        heatmap_img.save(heatmap_path)

        # superimpose the heatmap on original image
        self.logger.info(f"Creating superimposed image using heatmap...")
        superimposed_path = os.path.join(output_path, "superimposed.jpg")
        superimposed_arr = self.superimpose(original_img, heatmap_arr, alpha=1.0)
        superimposed_img = array_to_img(superimposed_arr)
        superimposed_img.save(superimposed_path)

        # superimpose the heatmap on original image with mask
        self.logger.info(f"Creating superimposed image with mask using heatmap...")
        masked_path = os.path.join(output_path, "masked.jpg")
        masked_arr = self.mask_superimpose(original_img, heatmap_arr)
        masked_img = array_to_img(masked_arr)
        masked_img.save(masked_path)

        return (
            self.class_names[predicted],
            heatmap_path,
            superimposed_path,
            masked_path
        )
