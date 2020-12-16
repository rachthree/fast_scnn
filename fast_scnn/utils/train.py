from pathlib import Path
import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter

from fast_scnn.utils import labels

# class ImageCallback(tf.keras.callbacks.Callback):
#     def __init__(self, log_dir, input_layer_name='input_layer', output_layer_name='output', **kwargs):
#         super().__init__(**kwargs)
#         self.log_dir = log_dir
#         self.img_dir = str(Path(log_dir, "images").mkdir(parents=True, exist_ok=True))
#         self.input_layer_name = input_layer_name
#         self.output_layer_name = output_layer_name
#
#     def on_train_begin(self, logs=None):
#         for i, layer in enumerate(self.model.layers):
#             if layer.name == self.input_layer_name:
#                 input_layer_ind = i
#
#             if layer.name == self.output_layer_name:
#                 output_layer_ind = i
#
#         self.input_layer_ind = input_layer_ind
#         self.output_layer_ind = output_layer_ind
#
#     def on_train_batch_end(self, batch, logs=None):
#         # in_batch = self.model.layers[self.input_layer_ind].input
#         # out_batch = self.model.layers[self.output_layer_ind].output
#
#         in_batch = tf.keras.backend.function(inputs=self.model.layers[self.input_layer_ind].input,
#                                              outputs=self.model.layers[self.input_layer_ind].output)
#         out_batch = tf.keras.backend.function(inputs=self.model.layers[self.output_layer_ind].input,
#                                               outputs=self.model.layers[self.output_layer_ind].output)
#
#         in_batch = in_batch.numpy()
#         out_batch = out_batch.numpy()
#
#         for i, (img, mask) in enumerate(zip(in_batch, out_batch)):
#             img_path = str(Path(self.img_dir, f"input_{i}.jpg"))
#             mask_path = str(Path(self.img_dir, f"label_{i}.jpg"))
#             tf.keras.preprocessing.image.save_img(img_path, (img * 255).astype(int))
#
#             mask = tf.argmax(mask, axis=-1)
#             mask_remap = labels.remap_mask_eval_id(mask)
#             mask_img = labels.id2rgb_seg_img(mask_remap, labels.label_colors)
#             tf.keras.preprocessing.image.save_img(mask_path, mask_img.astype(int))



def make_save_img_step(model, img_dir):
    train_step_o = model.train_step

    def save_img_step(data):
        # Train on data as usual
        result = train_step_o(data)

        # Save input and output images
        data = data_adapter.expand_1d(data)
        x, y_true, w = data_adapter.unpack_x_y_sample_weight(data)

        y_pred = model(x, training=False)

        in_batch = x.numpy()
        out_batch = y_pred[0].numpy()
        truth_batch = y_true['output'].numpy()

        for i, (img, mask, truth) in enumerate(zip(in_batch, out_batch, truth_batch)):
            img_path = str(Path(img_dir, f"input_{i}.jpg"))
            mask_path = str(Path(img_dir, f"label_{i}.jpg"))
            truth_path = str(Path(img_dir, f"truth_{i}.jpg"))
            tf.keras.preprocessing.image.save_img(img_path, (img * 255).astype(int))

            mask_img = labels.out_array_to_label_img(mask)
            tf.keras.preprocessing.image.save_img(mask_path, mask_img)

            truth_img = labels.out_array_to_label_img(truth)
            tf.keras.preprocessing.image.save_img(truth_path, truth_img)

        return result

    return save_img_step
