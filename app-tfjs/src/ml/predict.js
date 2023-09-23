import * as tf from '@tensorflow/tfjs';
import * as utils from "./utils"

const MODEL_URL = './model/model.json';

/** @type {tf.LayersModel} */

/**
 * Load TensorFlow.js model
 */
export async function loadModel() {
  return await tf.loadLayersModel(MODEL_URL)
}

/**
 * Load file as image
 * @param {Image} file 
 * @returns {tf.Tensor<tf.Rank>}
 */
export function loadImage(file) {
  return tf.browser.fromPixels(file)
    .toFloat()
    .resizeBilinear([160, 160])
    .div(tf.scalar(255))
    .reshape([1, 160, 160, 3]);
}

/**
 * Classify the input tensor
 * @param {tf.LayersModel} model
 * @param {tf.Tensor<tf.Rank>} x 
 * @returns {Promise<number[]>}
 */
export async function predict(model, x) {
  // execute the last layer
  const logits = model.predict(x);

  // softmax
  return Array.from(await tf.softmax(logits).data());
}

export async function getClass(proba) {
  // max class
  const predClass = await tf.argMax(proba).data();
  return predClass[0];
}

/**
 * Perform Grad-CAM on the input image for the classIndex
 * @param {tf.LayersModel} model
 * @param {tf.Tensor<tf.Rank>} x 
 * @param {number} classIndex 
 * @param {number} overlayFactor 
 * @returns 
 */
export function gradCAM(model, x, classIndex) {
  // Try to locate the last conv layer of the model.
  let layerIndex = model.layers.length - 1;
  while (layerIndex >= 0) {
    if (model.layers[layerIndex].getClassName().startsWith('Conv')) {
      break;
    }
    layerIndex--;
  }
  tf.util.assert(
    layerIndex >= 0, `Failed to find a convolutional layer in model`);

  const lastConvLayer = model.layers[layerIndex];
  console.log(
    `Located last convolutional layer of the model at ` +
    `index ${layerIndex}: layer type = ${lastConvLayer.getClassName()}; ` +
    `layer name = ${lastConvLayer.name}`);

  // Get "sub-model 1", which goes from the original input to the output
  // of the last convolutional layer.
  const lastConvLayerOutput = lastConvLayer.output;
  const subModel1 =
    tf.model({
      inputs: model.inputs,
      outputs: lastConvLayerOutput
    });

  // Get "sub-model 2", which goes from the output of the last convolutional
  // layer to the original output.
  const newInput = tf.input({ shape: lastConvLayerOutput.shape.slice(1) });
  layerIndex++;
  let y = newInput;
  while (layerIndex < model.layers.length) {
    y = model.layers[layerIndex++].apply(y);
  }
  const subModel2 = tf.model({ inputs: newInput, outputs: y });

  return tf.tidy(() => {
    // This function runs sub-model 2 and extracts the slice of the probability
    // output that corresponds to the desired class.
    const convOutput2ClassOutput = (input) =>
      subModel2.apply(input, { training: true }).gather([classIndex], 1);
    // This is the gradient function of the output corresponding to the desired
    // class with respect to its input (i.e., the output of the last
    // convolutional layer of the original model).
    const gradFunction = tf.grad(convOutput2ClassOutput);

    // Calculate the values of the last conv layer's output.
    const lastConvLayerOutputValues = subModel1.apply(x);
    // Calculate the values of gradients of the class output w.r.t. the output
    // of the last convolutional layer.
    const gradValues = gradFunction(lastConvLayerOutputValues);

    // Pool the gradient values within each filter of the last convolutional
    // layer, resulting in a tensor of shape [numFilters].
    const pooledGradValues = tf.mean(gradValues, [0, 1, 2]);
    // Scale the convlutional layer's output by the pooled gradients, using
    // broadcasting.
    const scaledConvOutputValues =
      lastConvLayerOutputValues.mul(pooledGradValues);

    // Create heat map by averaging and collapsing over all filters.
    let heatMap = scaledConvOutputValues.mean(-1);

    // Discard negative values from the heat map and normalize it to the [0, 1]
    // interval.
    heatMap = heatMap.relu();
    heatMap = heatMap.div(heatMap.max()).expandDims(-1);

    // Up-sample the heat map to the size of the input image.
    heatMap = tf.image.resizeBilinear(heatMap, [x.shape[1], x.shape[2]]);

    // Apply an RGB colormap on the heatMap. This step is necessary because
    // the heatMap is a 1-channel (grayscale) image. It needs to be converted
    // into a color (RGB) one through this function call.
    return utils.applyColorMap(heatMap);
  });
}

export async function superimposeImage(x, heatMap, overlayFactor = 2.0) {
  // To form the final output, overlay the color heat map on the input image.
  const superimposed = heatMap.mul(overlayFactor).add(x);
  return superimposed.div(superimposed.max()).mul(255);
}

/**
 * 
 * @param {tf.Tensor<tf.Rank>} x 
 * @param {tf.Tensor<tf.Rank>} heatmap 
 */
export async function maskImage(x, heatmap) {
  // calculate median from heatmap
  const threshold = x.mean();

  // create white image
  const whiteMat = tf.ones(x.shape);
  const heatmapRed = heatmap.slice([0, 0, 0, 0], [1, 160, 160, 1]);

  // cut the original image with 255 if heatmap is less than threshold, else fill original image
  const mask = tf.less(heatmapRed, threshold);
  const masked = tf.where(mask, whiteMat, x).reshape([1, 160, 160, 3]).mul(tf.scalar(255));
  
  return masked;
}

export async function getThreshold(heatmap) {
  const val = await tf.mean(heatmap).data();
  return val[0];
}
