import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# import tensorflow_addons as tfa


def ds_conv(input_layer, out_channels, kernel=(3, 3), strides=(1, 1), padding='same'):
    """
    Depthwise separable convolution with batchnorm between depthwise and pointwise convolutions, nonlinearity removed.

    Note: TF2 does not use batchnorm between DW and PW convolutions. MobileNet has ReLU after batchnorm.
    """
    x = layers.DepthwiseConv2D(kernel_size=kernel, strides=strides, padding=padding)(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(out_channels, kernel_size=(1, 1), strides=(1, 1), padding=padding,
                      kernel_regularizer=keras.regularizers.L2(l2=0.00004))(x)

    return x


def down_sample(input_layer):
    """Generates Fast-SCNN's downsampling module."""
    ds = layers.Conv2D(32, (3, 3), padding='same', strides=(2, 2),
                       kernel_regularizer=keras.regularizers.L2(l2=0.00004))(input_layer)
    ds = layers.BatchNormalization()(ds)
    ds = layers.Activation('relu')(ds)

    # ds = layers.SeparableConv2D(48, (3, 3), padding='same', strides=(2, 2))(ds)
    ds = ds_conv(ds, 48, kernel=(3, 3), strides=(2, 2), padding='same')
    ds = layers.BatchNormalization()(ds)
    ds = layers.Activation('relu')(ds)

    # ds = layers.SeparableConv2D(64, (3, 3), padding='same', strides=(2, 2))(ds)
    ds = ds_conv(ds, 64, kernel=(3, 3), strides=(2, 2), padding='same')
    ds = layers.BatchNormalization()(ds)
    ds = layers.Activation('relu')(ds)

    return ds


def _res_bottleneck(inputs, filters, kernel, t, s, residual=False):
    """Helper function for Fast-SCNN's bottleneck module."""
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
    """Generates Fast-SCNN's bottleneck module."""
    x = _res_bottleneck(inputs, filters, kernel, t, strides, residual=False)

    for i in range(1, n):
        x = _res_bottleneck(x, filters, kernel, t, 1, residual=True)

    return x


def pyramid_pooling_block(input_tensor, bin_sizes, reduce_dims=True):
    """Generates a pyramid pooling module a la PSPNet."""
    # Based on https://github.com/hszhao/semseg/blob/master/model/pspnet.py
    in_h, in_w = input_tensor.shape[1],  input_tensor.shape[2]
    concat_list = [input_tensor]
    n_filter_out = input_tensor.shape[-1] // len(bin_sizes) if reduce_dims else input_tensor.shape[-1] # 128 could've been incorrect

    for bin_size in bin_sizes:
        # PSP Paper and Pytorch implementation uses nn.AdaptiveAvgPool2d
        # https://stackoverflow.com/questions/58692476/what-is-adaptive-average-pooling-and-how-does-it-work
        # PSP uses PyTorch's nn.Conv2d default for padding (valid)
        # avg_pool_s = (in_h // bin_size, in_w // bin_size)
        # avg_pool_k = (in_h - (bin_size - 1) * avg_pool_s[0], in_w - (bin_size - 1) * avg_pool_s[1])
        # x = layers.AveragePooling2D(pool_size=avg_pool_k, strides=avg_pool_s, padding='valid')(input_tensor)
        # x = tfa.layers.AdaptiveAveragePooling2D(output_size=(bin_size, bin_size))(input_tensor)

        size = (in_h // bin_size, in_w // bin_size)
        x = layers.AveragePooling2D(pool_size=size, strides=size)(input_tensor)
        x = layers.Conv2D(n_filter_out, (1, 1), strides=(1, 1), padding='valid', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        # x = layers.experimental.preprocessing.Resizing(in_h, in_w, interpolation='bilinear')(x)

        x = layers.UpSampling2D(size=size, interpolation='bilinear')(x)

        concat_list.append(x)

    return layers.concatenate(concat_list)


def global_feature_extractor(lds_layer):
    """Generates Fast-SCNN's global feature extractor module"""
    gfe_layer = bottleneck_block(lds_layer, 64, (3, 3), t=6, strides=2, n=3)
    gfe_layer = bottleneck_block(gfe_layer, 96, (3, 3), t=6, strides=2, n=3)
    gfe_layer = bottleneck_block(gfe_layer, 128, (3, 3), t=6, strides=1, n=3)
    gfe_layer = pyramid_pooling_block(gfe_layer, [1, 2, 4, 8], reduce_dims=True)  # should be [1, 2, 3, 6] but does not divide nicely

    return gfe_layer


def feature_fusion(lds_layer, gfe_layer):
    """Generates Fast-SCNN's feature fusion module."""
    ff_layer1 = layers.Conv2D(128, (1, 1), padding='same', strides=(1, 1),
                              kernel_regularizer=keras.regularizers.L2(l2=0.00004))(lds_layer)
    ff_layer1 = layers.BatchNormalization()(ff_layer1)

    ff1_shape = ff_layer1.shape
    gfe_shape = gfe_layer.shape
    scale = (ff1_shape[1] // gfe_shape[1], ff1_shape[2] // gfe_shape[2])
    # ff_layer2 = layers.experimental.preprocessing.Resizing(ff1_shape[1], ff1_shape[2], interpolation='bilinear')(gfe_layer)
    ff_layer2 = layers.UpSampling2D(size=scale, interpolation='bilinear')(gfe_layer)
    ff_layer2 = layers.DepthwiseConv2D((3, 3), strides=1, padding='same', dilation_rate=scale)(ff_layer2)
    ff_layer2 = layers.BatchNormalization()(ff_layer2)
    ff_layer2 = layers.Activation('relu')(ff_layer2)
    ff_layer2 = layers.Conv2D(128, kernel_size=1, strides=1, padding='same', activation=None,
                              kernel_regularizer=keras.regularizers.L2(l2=0.00004))(ff_layer2)
    ff_layer2 = layers.BatchNormalization()(ff_layer2)

    ff_final = layers.add([ff_layer1, ff_layer2])
    ff_final = layers.Activation('relu')(ff_final)

    return ff_final


def classifier_layer(input_tensor, img_size, num_classes, name, resize_output=True):
    """Generates Fast-SCNN's classifier module."""
    # classifier = layers.SeparableConv2D(128, (3, 3), padding='same', strides=(1, 1))(input_tensor)
    classifier = ds_conv(input_tensor, 128, kernel=(3, 3), strides=(1, 1), padding='same')
    classifier = layers.BatchNormalization()(classifier)
    classifier = layers.Activation('relu')(classifier)

    # classifier = layers.SeparableConv2D(128, (3, 3), padding='same', strides=(1, 1))(classifier)
    classifier = ds_conv(classifier, 128, kernel=(3, 3), strides=(1, 1), padding='same')
    classifier = layers.BatchNormalization()(classifier)
    classifier = layers.Activation('relu')(classifier)

    classifier = layers.Dropout(0.3)(classifier)
    classifier = layers.Conv2D(num_classes, (1, 1), padding='same', strides=(1, 1),
                               kernel_regularizer=keras.regularizers.L2(l2=0.00004))(classifier)
    classifier = layers.BatchNormalization()(classifier)

    if resize_output:
        # classifier = layers.experimental.preprocessing.Resizing(img_size[0], img_size[1], interpolation='bilinear')(classifier)
        classifier = layers.UpSampling2D(size=(8, 8), interpolation='bilinear')(classifier)

    classifier = layers.Softmax(name=name)(classifier)

    return classifier


def aux_layer(input_tensor, num_classes, name, resize_aux_size=None):
    """
    Generates a auxiliary module for Fast-SCNN.

    Note: It is unknown what the authors used.
    """
    input_shape = input_tensor.shape
    in_channels = input_shape[-1]
    # aux = layers.SeparableConv2D(in_channels // 2, (3, 3), padding='same', strides=(1, 1))(input_tensor)  # 128
    aux = ds_conv(input_tensor, in_channels // 2, kernel=(3, 3), strides=(1, 1), padding='same')
    aux = layers.BatchNormalization()(aux)
    aux = layers.Activation('relu')(aux)
    #
    # # aux = layers.SeparableConv2D(128, (3, 3), padding='same', strides=(1, 1))(aux)
    # # aux = layers.BatchNormalization()(aux)
    # # aux = layers.Activation('relu')(aux)
    #
    aux = layers.Dropout(0.3)(aux)
    aux = layers.Conv2D(num_classes, (1, 1), padding='same', strides=(1, 1),
                        kernel_regularizer=keras.regularizers.L2(l2=0.00004))(aux)
    aux = layers.BatchNormalization()(aux)
    # input_shape = input_tensor.shape
    # in_channels = input_shape[-1]
    # aux = layers.Conv2D(in_channels // 2, (1, 1), padding='same', strides=(1, 1),
    #                     kernel_regularizer=keras.regularizers.L2(l2=0.00004))(input_tensor)
    # aux = layers.BatchNormalization()(aux)
    # aux = layers.Activation('relu')(aux)
    # aux = layers.Dropout(0.1)(aux)
    # aux = layers.Conv2D(num_classes, (1, 1), padding='same', strides=(1, 1),
    #                     kernel_regularizer=keras.regularizers.L2(l2=0.00004))(aux)


    if resize_aux_size is not None:
        aux_shape = aux.shape
        scale = (resize_aux_size[0] // aux_shape[1], resize_aux_size[1] // aux_shape[2])
        # aux = layers.experimental.preprocessing.Resizing(resize_aux_size[0], resize_aux_size[1], interpolation='bilinear')(aux)
        aux = layers.UpSampling2D(size=scale, interpolation='bilinear')(aux)

    aux = layers.Softmax(name=name)(aux)

    return aux
