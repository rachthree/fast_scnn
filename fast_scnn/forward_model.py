from pathlib import Path

import tensorflow as tf
import tensorflow.keras as keras

import numpy as np
# import cv2

class Predictor(object):
    def __init__(self, *, model_path, resize=None):
        model = keras.models.load_model(str(Path(model_path, 'checkpoints')))
        self.model = keras.Model(inputs=model.input, outputs=model.get_layer('output').output)
        self.resize = resize

    def __call__(self, img):
        img_tensor = tf.expand_dims(img, 0)
        output = self.model(img_tensor, training=False)
        mask = tf.argmax(output, axis=-1)
        mask = mask[0]

        results = {}
        if self.resize is not None:
            output_resized = tf.image.resize(output, self.resize, method=tf.image.ResizeMethod.BILINEAR)
            mask_resized = tf.argmax(output_resized, axis=-1)
            mask_resized = mask_resized[0]

            results['prob_resized'] = output_resized.numpy()
            results['mask_resized'] = mask_resized.numpy()

        results['output'] = output.numpy()
        results['mask'] = mask.numpy()

        return results


class Evaluator(object):
    def __init__(self, *, model_path, data_dir):
        model = keras.models.load_model(str(Path(model_path, 'checkpoints')))
        self.model = keras.Model(inputs=model.input, outputs=model.get_layer('output').output)
        self.data_dir = data_dir

    def evaluate(self):
        pass
