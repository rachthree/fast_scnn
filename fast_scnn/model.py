import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from fast_scnn.utils import layers


def generate_model(n_classes, ds_aux_weight=0.4, gfe_aux_weight=0.4, summary=False, mode='train'):
    tf.config.run_functions_eagerly(True)
    if mode == 'train':
        h, w = None, None
    elif mode == 'debug':
        h, w = 1024, 2048
    else:
        raise ValueError(f"Mode {mode} not recognized.")

    img_shape = (h, w, 3)

    input_layer = tf.keras.layers.Input(shape=img_shape, name='input_layer')
    input_shape = tf.shape(input_layer)

    gaus_layer = tf.keras.layers.GaussianNoise(stddev=0.02)(input_layer)
    ds_layer = layers.down_sample(gaus_layer)
    gfe_layer = layers.global_feature_extractor(ds_layer)
    ff_final = layers.feature_fusion(ds_layer, gfe_layer)

    ds_shape = tf.shape(ds_layer)
    classifier = layers.classifier_layer(ff_final, (input_shape[1], input_shape[2]), n_classes, name='output')
    ds_aux = layers.classifier_layer(ds_layer, (input_shape[1], input_shape[2]), n_classes, name='ds_aux')
    gfe_aux = layers.classifier_layer(gfe_layer, (input_shape[1], input_shape[2]), n_classes, name='gfe_aux', resize_input=(ds_shape[1], ds_shape[2]))
    # classifier = layers.ClassifierBlock(n_classes=n_classes, name='output')(ff_final, input_layer)
    # ds_aux = layers.ClassifierBlock(n_classes=n_classes, name='ds_aux')(ds_layer, input_layer)
    # gfe_aux = layers.ClassifierBlock(n_classes=n_classes, name='gfe_aux')(ds_layer, input_layer, base_resize_layer=ds_layer)

    loss_dict = {'output': 'categorical_crossentropy',
                 'ds_aux': 'categorical_crossentropy',
                 'gfe_aux': 'categorical_crossentropy'}
    loss_weights = {'output': 1.0,
                    'ds_aux': ds_aux_weight,
                    'gfe_aux': gfe_aux_weight}

    model = keras.Model(inputs=input_layer, outputs=[classifier, ds_aux, gfe_aux], name='Fast_SCNN')
    if summary:
        model.summary()
        tf.keras.utils.plot_model(model, 'fast_scnn.png', show_shapes=True, show_layer_names=True)

    return model, loss_dict, loss_weights
