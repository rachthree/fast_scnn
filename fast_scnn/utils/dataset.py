from pathlib import Path
import numpy as np
import tensorflow as tf

from fast_scnn.utils.labels import labels, keep_labels

AUTO = tf.data.experimental.AUTOTUNE


class CityScapesDataset(object):
    def __init__(self, *, data_dir, label_dir, prefetch=1, batch_size=32, seed=None, num_parallel_calls=1, img_norm=True,
                 output_names=None, resize_aux=None, augment=False, autotune=False, data_suffix='_leftImg8bit', label_suffix='_gtFine_labelIds'):
        self.data_dir = data_dir
        self.data_suffix = data_suffix
        self.label_dir = label_dir
        self.label_suffix = label_suffix
        self.prefetch = prefetch
        self.batch_size = batch_size
        self.num_parallel_calls = num_parallel_calls
        self.seed = seed
        self.img_norm = img_norm
        self.output_names = output_names
        self.augment = augment
        self.resize_aux = resize_aux

        if self.seed is not None:
            np.random.seed(self.seed)

        if autotune:
            self.prefetch = AUTO
            self.num_parallel_calls = AUTO

        # Random color channels noise - only to image - hue, saturation, contrast?
        # Random brightness - only to image
        self.color_ops = [lambda img: tf.image.random_hue(img, 0.2, seed=seed),
                          lambda img: tf.image.random_saturation(img, lower=0.5, upper=1.5, seed=seed),
                          lambda img: tf.image.random_contrast(img, lower=0.5, upper=1.5, seed=seed),
                          lambda img: tf.image.random_brightness(img, max_delta=32./255., seed=seed)]

    def parse_data(self, filepath):
        label_filepath = tf.strings.regex_replace(filepath, self.data_dir.replace("\\", "\\\\"), self.label_dir.replace("\\", "\\\\"))
        label_filepath = tf.strings.regex_replace(label_filepath, self.data_suffix, self.label_suffix)

        img_str = tf.io.read_file(filepath)
        img = tf.image.decode_png(img_str, channels=0)
        img = img[..., :3]
        img = tf.cast(img, tf.float32)
        img_shape = tf.shape(img)

        label_str = tf.io.read_file(label_filepath)
        label = tf.image.decode_png(label_str, channels=0)
        label = label[..., 0]
        label = tf.reshape(label, (img_shape[0], img_shape[1]))  # tf.squeeze results in unknown shape
        label = tf.cast(label, tf.int32)

        return img, label

    def preprocess(self, img, label):
        if self.img_norm:
            img /= 255.0

        label = tf.one_hot(label, depth=len(labels))
        label = tf.boolean_mask(label, np.array(keep_labels), axis=2)  # only train for labels used for evaluation

        # Per image augmentation
        # Horizontal flip
        if self.augment and tf.random.uniform([]) > 0.5:
            img = tf.image.flip_left_right(img)
            label = tf.image.flip_left_right(label)

        return img, label

    def augment_batch(self, img_batch, output_batch):
        img_batch_shape = tf.shape(img_batch)
        n = img_batch_shape[0]
        h = tf.cast(img_batch_shape[1], tf.float32)
        w = tf.cast(img_batch_shape[2], tf.float32)

        # Random resizing - both image and label
        scale = tf.random.uniform([], minval=0.5, maxval=2.0, dtype=tf.float32, seed=self.seed)
        resize_shape = tf.cast([scale*h, scale*w], tf.int32)
        img_aug = tf.image.resize(img_batch, resize_shape, method=tf.image.ResizeMethod.BILINEAR,
                                  preserve_aspect_ratio=True)
        label_aug = tf.image.resize(output_batch, resize_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                                    preserve_aspect_ratio=True)

        # Random translation/crop - both image and label

        if scale > 1.0:
            # crop to input shape
            io_cat = tf.concat([img_batch, output_batch], axis=-1)
            n_dim = tf.shape(io_cat)[-1]

            io_cat = tf.image.random_crop(io_cat, (n, img_batch_shape[1], img_batch_shape[2], n_dim), seed=self.seed)
            img_aug, label_aug = io_cat[..., :3], io_cat[..., 3:]

        elif scale < 1.0:
            # resize using padding
            img_aug = tf.image.resize_with_crop_or_pad(img_batch, img_batch_shape[1], img_batch_shape[2])
            label_aug = tf.image.resize_with_crop_or_pad(output_batch, img_batch_shape[1], img_batch_shape[2])


        # GaussianNoise layer to be added to model directly
        # Order of operations does matter... randomize operations
        np.random.shuffle(self.color_ops)
        for op in self.color_ops:
            img_aug = op(img_aug)

        return img_aug, label_aug

    def output_aux(self, img_batch, output_batch):
        out_dict = {'output': output_batch}
        for layer in self.output_names:
            if layer != 'output':
                out_dict[layer] = tf.image.resize(output_batch, (self.resize_aux[layer][0], self.resize_aux[layer][1]),
                                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                                                  preserve_aspect_ratio=True)

        return img_batch, out_dict

    def generate_dataset(self):
        files = tf.io.match_filenames_once(str(Path(self.data_dir, '*/*.png')))
        dataset_size = tf.cast(tf.shape(files)[0], tf.int64).numpy()

        dataset = tf.data.Dataset.from_tensor_slices(files)
        dataset = dataset.shuffle(buffer_size=dataset_size, seed=self.seed)
        dataset = dataset.map(self.parse_data, num_parallel_calls=self.num_parallel_calls)
        dataset = dataset.map(self.preprocess, num_parallel_calls=self.num_parallel_calls)
        dataset = dataset.batch(self.batch_size)
        if self.augment:
            dataset = dataset.map(self.augment_batch, num_parallel_calls=self.num_parallel_calls)
        if self.output_names is not None:
            dataset = dataset.map(self.output_aux, num_parallel_calls=self.num_parallel_calls)
        dataset = dataset.prefetch(self.prefetch)
        return dataset, dataset_size
