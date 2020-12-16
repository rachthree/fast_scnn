import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from fast_scnn.utils import layers


def generate_model(n_classes, img_size=(1024, 2048), input_size_factor=None,
                   ds_aux_weight=0.4, gfe_aux_weight=0.4, summary=False,
                   resize_output=True, resize_aux=True, ds_aux=True, gfe_aux=True):
    h, w = img_size
    if input_size_factor is not None:
        h = int(h * input_size_factor)
        w = int(w * input_size_factor)

    img_shape = (h, w, 3)

    input_layer = tf.keras.layers.Input(shape=img_shape, name='input_layer')
    input_shape = input_layer.shape

    gaus_layer = tf.keras.layers.GaussianNoise(stddev=0.02)(input_layer)
    ds_layer = layers.down_sample(gaus_layer)
    gfe_layer = layers.global_feature_extractor(ds_layer)
    ff_final = layers.feature_fusion(ds_layer, gfe_layer)

    classifier = layers.classifier_layer(ff_final, (input_shape[1], input_shape[2]), n_classes,
                                         name='output', resize_output=resize_output)

    resize_aux_size = (h, w) if resize_aux else None
    loss_dict = {'output': 'categorical_crossentropy'}
    loss_weights = {'output': 1.0}
    outputs = [classifier]

    if ds_aux:
        ds_aux = layers.aux_layer(ds_layer, n_classes, name='ds_aux', resize_aux_size=resize_aux_size)
        loss_dict['ds_aux'] = 'categorical_crossentropy'
        loss_weights['ds_aux'] = ds_aux_weight
        outputs.append(ds_aux)

    if gfe_aux:
        gfe_aux = layers.aux_layer(gfe_layer, n_classes, name='gfe_aux', resize_aux_size=resize_aux_size)
        loss_dict['gfe_aux'] = 'categorical_crossentropy'
        loss_weights['gfe_aux'] = gfe_aux_weight
        outputs.append(gfe_aux)

    model = keras.Model(inputs=input_layer, outputs=outputs, name='Fast_SCNN')
    if summary:
        model.summary()
        tf.keras.utils.plot_model(model, 'fast_scnn.png', show_shapes=True, show_layer_names=True)

    return model, loss_dict, loss_weights
