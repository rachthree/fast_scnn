import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, activations
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


def pyramid_pooling_block(input_tensor, bin_sizes, in_h, in_w, reduce_dims=True):
    # Based on https://github.com/hszhao/semseg/blob/master/model/pspnet.py
    concat_list = [input_tensor]
    n_filter_out = input_tensor.shape[-1] // len(bin_sizes) if reduce_dims else input_tensor.shape[-1] # 128 could've been incorrect

    for bin_size in bin_sizes:
        # Paper and Pytorch implementation uses nn.AdaptiveAvgPool2d
        # https://stackoverflow.com/questions/58692476/what-is-adaptive-average-pooling-and-how-does-it-work
        # avg_pool_s = (in_h // bin_size, in_w // bin_size)
        # avg_pool_k = (in_h - (bin_size - 1) * avg_pool_s[0], in_w - (bin_size - 1) * avg_pool_s[1])
        # x = layers.AveragePooling2D(pool_size=avg_pool_k, strides=avg_pool_s, padding='valid')(input_tensor)

        # x = layers.AveragePooling2D(pool_size=(in_h // bin_size, in_w // bin_size),
        #                             strides=(in_h // bin_size, in_w // bin_size))(input_tensor)
        x = tfa.layers.AdaptiveAveragePooling2D(output_size=(bin_size, bin_size))(input_tensor)
        x = layers.Conv2D(n_filter_out, (1, 1), strides=(1, 1), padding='valid', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        # x = layers.Lambda(lambda x_in: tf.image.resize(x_in, (in_h, in_w), method=tf.image.ResizeMethod.BILINEAR))(x)
        x = layers.experimental.preprocessing.Resizing(in_h, in_w, interpolation='bilinear')(x)

        concat_list.append(x)

    return layers.concatenate(concat_list)


def global_feature_extractor(lds_layer):
    gfe_layer = bottleneck_block(lds_layer, 64, (3, 3), t=6, strides=2, n=3)
    gfe_layer = bottleneck_block(gfe_layer, 96, (3, 3), t=6, strides=2, n=3)
    gfe_layer = bottleneck_block(gfe_layer, 128, (3, 3), t=6, strides=1, n=3)
    gfe_shape = tf.shape(gfe_layer)
    gfe_layer = pyramid_pooling_block(gfe_layer, [1, 2, 3, 6], gfe_shape[1], gfe_shape[2], reduce_dims=True)  # 2 4 6 8?

    return gfe_layer


def feature_fusion(lds_layer, gfe_layer):
    ff_layer1 = layers.Conv2D(128, (1, 1), padding='same', strides=(1, 1),
                              kernel_regularizer=keras.regularizers.L2(l2=0.00004))(lds_layer)
    # ff_layer1 = layers.BatchNormalization()(ff_layer1)
    # ff_layer1 = layers.Activation('relu')(ff_layer1)

    # ff_layer2 = layers.UpSampling2D((4, 4), interpolation='bilinear')(gfe_layer)
    ff1_shape = tf.shape(ff_layer1)
    # gfe_shape = tf.shape(gfe_layer).numpy()
    # scale = (ff_layer1.shape[1] // gfe_layer.shape[1], ff_layer1.shape[2] // gfe_layer.shape[2])
    # scale = (ff1_shape[1] // gfe_shape[1], ff1_shape[2] // gfe_shape[2])
    # ff_layer2 = layers.Lambda(lambda x_in: tf.image.resize(x_in, (ff_layer1.shape[1], ff_layer1.shape[2]),
    #                                                        method=tf.image.ResizeMethod.BILINEAR))(gfe_layer)
    ff_layer2 = layers.experimental.preprocessing.Resizing(ff1_shape[1], ff1_shape[2], interpolation='bilinear')(gfe_layer)
    ff_layer2 = layers.DepthwiseConv2D((3, 3), strides=1, padding='same', dilation_rate=(4,4))(ff_layer2)  # dilation_rate = (4,4)) or scale
    ff_layer2 = layers.BatchNormalization()(ff_layer2)
    ff_layer2 = layers.Activation('relu')(ff_layer2)
    ff_layer2 = layers.Conv2D(128, kernel_size=1, strides=1, padding='same', activation=None,
                              kernel_regularizer=keras.regularizers.L2(l2=0.00004))(ff_layer2)

    ff_final = layers.add([ff_layer1, ff_layer2])
    ff_final = layers.BatchNormalization()(ff_final)
    ff_final = layers.Activation('relu')(ff_final)

    return ff_final


def classifier_layer(input_tensor, img_size, num_classes, name, resize_input=None):
    if resize_input is not None:
        # input_tensor = layers.Lambda(lambda x_in: tf.image.resize(x_in, (resize_input[0], resize_input[1]),
        #                                                           method=tf.image.ResizeMethod.BILINEAR))(input_tensor)
        input_tensor = layers.experimental.preprocessing.Resizing(resize_input[0], resize_input[1], interpolation='bilinear')(input_tensor)
    classifier = layers.SeparableConv2D(128, (3, 3), padding='same', strides=(1, 1))(input_tensor)
    classifier = layers.BatchNormalization()(classifier)
    classifier = layers.Activation('relu')(classifier)

    classifier = layers.SeparableConv2D(128, (3, 3), padding='same', strides=(1, 1))(classifier)
    classifier = layers.BatchNormalization()(classifier)
    classifier = layers.Activation('relu')(classifier)

    classifier = layers.Conv2D(num_classes, (1, 1), padding='same', strides=(1, 1),
                               kernel_regularizer=keras.regularizers.L2(l2=0.00004))(classifier)
    classifier = layers.BatchNormalization()(classifier)
    classifier = layers.Activation('relu')(classifier)

    # classifier = layers.Lambda(lambda x_in: tf.image.resize(x_in, (img_size[0], img_size[1]),
    #                                                         method=tf.image.ResizeMethod.BILINEAR))(classifier)
    classifier = layers.experimental.preprocessing.Resizing(img_size[0], img_size[1], interpolation='bilinear')(classifier)
    classifier = layers.Dropout(0.3)(classifier)
    classifier = layers.Softmax()(classifier)
    classifier = layers.Activation('relu', name=name)(classifier)

    return classifier

class ClassifierBlock(keras.layers.Layer):
    def __init__(self, *args, n_classes, **kwargs):
        super(ClassifierBlock, self).__init__(*args, **kwargs)

        # resize here
        self.sepconv2d1 = layers.SeparableConv2D(128, (3, 3), padding='same', strides=(1, 1))
        self.batch_norm1 = layers.BatchNormalization()
        self.act_relu1 = layers.Activation('relu')

        self.sepconv2d2 = layers.SeparableConv2D(128, (3, 3), padding='same', strides=(1, 1))
        self.batch_norm2 = layers.BatchNormalization()
        self.act_relu2 = layers.Activation('relu')

        self.conv2d = layers.Conv2D(n_classes, (1, 1), padding='same', strides=(1, 1),
                                    kernel_regularizer=keras.regularizers.L2(l2=0.00004))
        self.batch_norm3 = layers.BatchNormalization()
        self.act_relu3 = layers.Activation('relu')

        # resize here

        self.dropout = layers.Dropout(0.3)
        self.softmax = layers.Softmax()
        self.act_relu4 = layers.Activation('relu')

    def call(self, prev_layer, input_layer, base_resize_layer=None, training=False):
        layer1 = prev_layer
        input_layer_shape = tf.shape(input_layer)
        if base_resize_layer is not None:
            base_layer_shape = tf.shape(base_resize_layer)
            layer1 = layers.experimental.preprocessing.Resizing(base_layer_shape[1], base_layer_shape[2],
                                                                interpolation='bilinear')(layer1)
        classifier = self.sepconv2d1(layer1)
        classifier = self.batch_norm1(classifier, training=training)
        classifier = self.act_relu1(classifier)

        classifier = self.sepconv2d2(classifier)
        classifier = self.batch_norm2(classifier, training=training)
        classifier = self.act_relu2(classifier)

        classifier = self.conv2d(classifier)
        classifier = self.batch_norm3(classifier)
        classifier = self.act_relu3(classifier)

        classifier = layers.experimental.preprocessing.Resizing(input_layer_shape[1], input_layer_shape[2],
                                                                interpolation='bilinear')(classifier)
        classifier = self.dropout(classifier, training=training)
        classifier = self.softmax(classifier)
        classifier = self.act_relu4(classifier)

        return classifier
