import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from fast_scnn.utils import layers


def generate_model(n_classes, img_size=(1024, 2048), ds_aux_weight=0.4, gfe_aux_weight=0.4, summary=False):
    h, w = img_size

    img_shape = (h, w, 3)

    input_layer = tf.keras.layers.Input(shape=img_shape, name='input_layer')
    input_shape = input_layer.shape   # tf.shape(input_layer)

    gaus_layer = tf.keras.layers.GaussianNoise(stddev=0.02)(input_layer)
    ds_layer = layers.down_sample(gaus_layer)
    gfe_layer = layers.global_feature_extractor(ds_layer)
    ff_final = layers.feature_fusion(ds_layer, gfe_layer)

    classifier = layers.classifier_layer(ff_final, (input_shape[1], input_shape[2]), n_classes, name='output')
    ds_aux = layers.aux_layer(ds_layer, n_classes, name='ds_aux')
    gfe_aux = layers.aux_layer(gfe_layer, n_classes, name='gfe_aux')

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