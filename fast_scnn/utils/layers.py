import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa


def down_sample(input_layer):
    ds = layers.Conv2D(32, (3, 3), padding='same', strides=(2, 2),
                       kernel_regularizer=keras.regularizers.L2(l2=0.00004))(input_layer)
    ds = layers.BatchNormalization()(ds)
    ds = layers.Activation('relu')(ds)

    ds = layers.SeparableConv2D(48, (3, 3), padding='same', strides=(2, 2))(ds)
    ds = layers.BatchNormalization()(ds)
    ds = layers.Activation('relu')(ds)

    ds = layers.SeparableConv2D(64, (3, 3), padding='same', strides=(2, 2))(ds)
    ds = layers.BatchNormalization()(ds)
    ds = layers.Activation('relu')(ds)

    return ds


def _res_bottleneck(inputs, filters, kernel, t, s, residual=False):
    t_channel = keras.backend.int_shape(inputs)[-1] * t

    x = layers.Conv2D(t_channel, (1, 1), padding='same', strides=(1, 1),
                      kernel_regularizer=keras.regularizers.L2(l2=0.00004))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters, (1, 1), padding='same', strides=(1, 1),
                      kernel_regularizer=keras.regularizers.L2(l2=0.00004))(x)
    x = layers.BatchNormalization()(x)

    if residual:
        x = layers.add([x, inputs])
    return x


def bottleneck_block(inputs, filters, kernel, t, strides, n):
    x = _res_bottleneck(inputs, filters, kernel, t, strides, residual=False)

    for i in range(1, n):
        x = _res_bottleneck(x, filters, kernel, t, 1, residual=True)

    return x


def pyramid_pooling_block(input_tensor, bin_sizes, reduce_dims=True):
    # Based on https://github.com/hszhao/semseg/blob/master/model/pspnet.py
    in_shape = tf.shape(input_tensor)
    # in_h, in_w = in_shape[1],  in_shape[2]

    concat_list = [input_tensor]
    n_filter_out = input_tensor.shape[-1] // len(bin_sizes) if reduce_dims else input_tensor.shape[-1] # 128 could've been incorrect

    for bin_size in bin_sizes:
        # Paper and Pytorch implementation uses nn.AdaptiveAvgPool2d
        # https://stackoverflow.com/questions/58692476/what-is-adaptive-average-pooling-and-how-does-it-work
        # avg_pool_s = (in_h // bin_size, in_w // bin_size)
        # avg_pool_k = (in_h - (bin_size - 1) * avg_pool_s[0], in_w - (bin_size - 1) * avg_pool_s[1])
        # x = layers.AveragePooling2D(pool_size=avg_pool_k, strides=avg_pool_s, padding='valid')(input_tensor)

        # Resize input to be divisible by the bin bin size
        new_h = find_divisible_bin_size(in_shape[1], bin_size)
        new_w = find_divisible_bin_size(in_shape[2], bin_size)
        x = AdaptiveResize(interpolation='bilinear')(x, (new_h, new_w))

        x = tfa.layers.AdaptiveAveragePooling2D(output_size=(bin_size, bin_size))(input_tensor)

        # pool_stride_size = in_shape[1:3] // bin_size
        # pool_stride_size = pool_stride_size.eval()
        # x = layers.AveragePooling2D(pool_size=pool_stride_size, strides=pool_stride_size)(input_tensor)
        x = layers.Conv2D(n_filter_out, (1, 1), strides=(1, 1), padding='valid', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        # x = layers.experimental.preprocessing.Resizing(in_h, in_w, interpolation='bilinear')(x)
        x = AdaptiveResize(interpolation='bilinear')(x, in_shape[1:3])

        concat_list.append(x)

    return layers.concatenate(concat_list)


def global_feature_extractor(lds_layer):
    gfe_layer = bottleneck_block(lds_layer, 64, (3, 3), t=6, strides=2, n=3)
    gfe_layer = bottleneck_block(gfe_layer, 96, (3, 3), t=6, strides=2, n=3)
    gfe_layer = bottleneck_block(gfe_layer, 128, (3, 3), t=6, strides=1, n=3)
    gfe_layer = pyramid_pooling_block(gfe_layer, [1, 2, 3, 6], reduce_dims=True)

    return gfe_layer


def feature_fusion(lds_layer, gfe_layer):
    ff_layer1 = layers.Conv2D(128, (1, 1), padding='same', strides=(1, 1),
                              kernel_regularizer=keras.regularizers.L2(l2=0.00004))(lds_layer)

    ff1_shape = tf.shape(ff_layer1)
    gfe_shape = tf.shape(gfe_layer)
    scale = tf.divide(ff1_shape[1:3], gfe_shape[1:3])
    # scale = (ff1_shape[1] // gfe_shape[1], ff1_shape[2] // gfe_shape[2])
    # ff_layer2 = layers.experimental.preprocessing.Resizing(ff1_shape[1], ff1_shape[2], interpolation='bilinear')(gfe_layer)
    ff_layer2 = AdaptiveResize(interpolation='bilinear')(gfe_layer, ff1_shape[1:3])
    ff_layer2 = layers.DepthwiseConv2D((3, 3), strides=1, padding='same', dilation_rate=scale)(ff_layer2)
    ff_layer2 = layers.BatchNormalization()(ff_layer2)
    ff_layer2 = layers.Activation('relu')(ff_layer2)
    ff_layer2 = layers.Conv2D(128, kernel_size=1, strides=1, padding='same', activation=None,
                              kernel_regularizer=keras.regularizers.L2(l2=0.00004))(ff_layer2)

    ff_final = layers.add([ff_layer1, ff_layer2])
    ff_final = layers.BatchNormalization()(ff_final)
    ff_final = layers.Activation('relu')(ff_final)

    return ff_final


def classifier_layer(input_tensor, num_classes, name, output_resize=None):
    classifier = layers.SeparableConv2D(128, (3, 3), padding='same', strides=(1, 1))(input_tensor)
    classifier = layers.BatchNormalization()(classifier)
    classifier = layers.Activation('relu')(classifier)

    classifier = layers.SeparableConv2D(128, (3, 3), padding='same', strides=(1, 1))(classifier)
    classifier = layers.BatchNormalization()(classifier)
    classifier = layers.Activation('relu')(classifier)

    classifier = layers.Dropout(0.3)(classifier)
    classifier = layers.Conv2D(num_classes, (1, 1), padding='same', strides=(1, 1),
                               kernel_regularizer=keras.regularizers.L2(l2=0.00004))(classifier)
    classifier = layers.BatchNormalization()(classifier)

    if output_resize is not None:
        # classifier = layers.experimental.preprocessing.Resizing(img_size[0], img_size[1], interpolation='bilinear')(classifier)
        classifier = AdaptiveResize(interpolation='bilinear')(classifier, output_resize)

    classifier = layers.Softmax(name=name)(classifier)

    return classifier


def aux_layer(input_tensor, num_classes, name, aux_resize=None):
    aux = layers.SeparableConv2D(128, (3, 3), padding='same', strides=(1, 1))(input_tensor)
    aux = layers.BatchNormalization()(aux)
    aux = layers.Activation('relu')(aux)

    # aux = layers.SeparableConv2D(128, (3, 3), padding='same', strides=(1, 1))(aux)
    # aux = layers.BatchNormalization()(aux)
    # aux = layers.Activation('relu')(aux)

    aux = layers.Dropout(0.3)(aux)
    aux = layers.Conv2D(num_classes, (1, 1), padding='same', strides=(1, 1),
                        kernel_regularizer=keras.regularizers.L2(l2=0.00004))(aux)
    aux = layers.BatchNormalization()(aux)

    if aux_resize is not None:
        # aux = layers.experimental.preprocessing.Resizing(resize_aux_size[0], resize_aux_size[1], interpolation='bilinear')(aux)
        aux = AdaptiveResize(interpolation='bilinear')(aux, aux_resize)

    aux = layers.Softmax(name=name)(aux)

    return aux


class AdaptiveResize(tf.keras.layers.Layer):
    def __init__(self, *, interpolation="bilinear", **kwargs):
        super().__init__(**kwargs)
        self.interpolation = interpolation
        self._interpolation_method = tf.python.keras.layers.preprocessing.image_preprocessing.get_interpolation(interpolation)

    def call(self, inputs, resize_shape):
        outputs = tf.python.ops.image_ops.resize_images_v2(images=inputs,
                                                           size=resize_shape,
                                                           method=self._interpolation_method)
        return outputs

    # def compute_output_shape(self, input_shape):
    #     input_shape = tf.python.framework.tensor_shape.TensorShape(input_shape).as_list()
    #     return tf.python.framework.tensor_shape.TensorShape([input_shape[0], self.resize_shape[0], self.resize_shape[1], input_shape[3]])

    def get_config(self):
        config = {'interpolation': self.interpolation}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf.function
def find_divisible_bin_size(dim_len, bin_size):
    if dim_len <= 0:
        raise ValueError(f"dim_len most be positive! Got {dim_len}")

    # Based on finding closest number to n that is divisible by m
    # Find the quotient
    q = dim_len // bin_size

    # 1st possible closest number
    n1 = bin_size * q

    # 2nd possible closest number
    n2 = (bin_size * (q + 1))

    # if true, then n1 is the required closest number
    if abs(dim_len - n1) < abs(dim_len - n2):
        result = (n1, n2)
    else:
        result = (n2, n1)

    return result[0] if result[0] > 0 else result[1]
