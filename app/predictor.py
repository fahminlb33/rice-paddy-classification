import os
from typing import Tuple, List

import joblib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf

class PredictorModel:
  tf_model: tf.keras.models.Model = None
  class_names: List[str] = None
  initialized: bool = False
  IMG_SIZE: tuple[int, int] = (160, 160)

  def load_model(self, model_root: str, tf_name: str, class_name: str) -> None:
    self.tf_model = tf.keras.models.load_model(os.path.join(model_root, tf_name))
    self.class_names = joblib.load(os.path.join(model_root, class_name))
    self.initialized = True

  def make_gradcam_heatmap(self, img_array, last_conv_layer_name: str, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    convLayer = self.tf_model.get_layer(last_conv_layer_name).output
    grad_model = tf.keras.models.Model(
        [self.tf_model.inputs], [convLayer, self.tf_model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_logits, dense_output_logits = grad_model(img_array)
        pred_index = tf.argmax(dense_output_logits[0])
        loss_value = dense_output_logits[:, pred_index]

        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        grads = tape.gradient(loss_value, last_conv_layer_logits)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        # then sum all the channels to obtain the heatmap class activation
        heatmap = tf.matmul(last_conv_layer_logits[0], pooled_grads[..., tf.newaxis])
        heatmap = tf.squeeze(heatmap)

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()

  def predict(self, image_path: str, output_path: str, alpha: float = 0.4) -> Tuple[str, str]:
    if not self.initialized:
      raise Exception("Predictor model not initialized")

    # load image
    image_tensor = tf.keras.preprocessing.image.load_img(image_path, target_size=self.IMG_SIZE)
    image_tensor = tf.keras.preprocessing.image.img_to_array(image_tensor)
    image_tensor = tf.keras.applications.mobilenet.preprocess_input(image_tensor)
    image_tensor = tf.expand_dims(image_tensor, axis=0)

    # predict
    prediction = self.tf_model.predict(image_tensor)
    prediction = tf.nn.softmax(prediction)
    prediction = tf.argmax(prediction, axis=1)

    # Rescale heatmap to a range 0-255
    heatmap = self.make_gradcam_heatmap(image_tensor, "Conv_1")
    heatmap = np.uint8(255 * heatmap)

    # Use RGB values of the colormap
    nipy_cm = plt.cm.get_cmap("nipy_spectral")
    nipy_colors = nipy_cm(np.arange(256))[:, :3]
    cm_heatmap = nipy_colors[heatmap]

    # load original image
    original_img = tf.keras.preprocessing.image.load_img(image_path)
    original_img = tf.keras.preprocessing.image.img_to_array(original_img)

    # Create an image with RGB colorized heatmap
    cm_heatmap = tf.keras.preprocessing.image.array_to_img(cm_heatmap)
    cm_heatmap = cm_heatmap.resize((original_img.shape[1], original_img.shape[0]))
    cm_heatmap = tf.keras.preprocessing.image.img_to_array(cm_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = cm_heatmap * alpha + original_img
    superimposed_img: Image = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    superimposed_img.save(output_path)

    return (self.class_names[prediction[0]], output_path)
