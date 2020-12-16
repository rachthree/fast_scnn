from pathlib import Path
import time
from argparse import ArgumentParser

import tensorflow as tf
from tensorflow import keras

from fast_scnn.utils.config import load_config
from fast_scnn.utils.dataset import CityScapesDataset
from fast_scnn.utils.labels import eval_labels
from fast_scnn.utils.metrics import OneHotMeanIoU
from fast_scnn.utils.train import make_save_img_step
from fast_scnn.model import generate_model


class Trainer(object):
    def __init__(self, *, train_dir, train_label_dir, val_dir, val_label_dir, save_dir,
                 sess_name, autotune_dataset=False, prefetch=1, num_parallel_calls=1,
                 input_names, output_names, ds_aux=True, gfe_aux=True, resize_aux_label=None, float_type,
                 epochs, early_stopping, seed=None, start_learning_rate, end_learning_rate, batch_size=12,
                 save_train_images=False,
                 resize_label=False,
                 input_size_factor=None):

        self.save_dir = Path(save_dir, sess_name)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.save_dir = str(self.save_dir)

        self.img_dir = Path(save_dir, sess_name, "images")
        self.img_dir.mkdir(exist_ok=True, parents=True)
        self.img_dir = str(self.img_dir)

        self.input_layer_names = input_names
        if not ds_aux:
            output_names.remove('ds_aux')
        if not gfe_aux:
            output_names.remove('gfe_aux')
        self.output_layer_names = output_names

        self.epochs = epochs
        self.early_stopping = early_stopping
        self.seed = seed
        self.start_learning_rate = start_learning_rate
        self.end_learning_rate = end_learning_rate
        self.save_train_images = save_train_images

        self.train_dir = train_dir
        self.train_label_dir = train_label_dir
        self.val_dir = val_dir
        self.val_label_dir = val_label_dir
        self.batch_size = batch_size
        self.resize_aux_label = resize_aux_label
        self.resize_label = resize_label
        self.input_size_factor = input_size_factor
        self.float_type = float_type
        self.autotune_dataset = autotune_dataset
        self.prefetch = prefetch
        self.num_parallel_calls = num_parallel_calls

        self.sess_name = time.ctime(time.time()).replace(':', '.') if sess_name is None else sess_name

        self.resize_output = not (isinstance(resize_label, list) or isinstance(resize_label, tuple))
        self.ds_aux = ds_aux
        self.gfe_aux = gfe_aux
        self.resize_aux = not isinstance(resize_aux_label, dict)
        self.n_classes = len(eval_labels)

    def get_dataset(self, output_sizes=None):
        datasets = {}
        print('\nPrepping training dataset...\n')
        datasets['train'], datasets['n_train'] = CityScapesDataset(data_dir=str(Path(self.train_dir)), label_dir=str(Path(self.train_label_dir)),
                                                                   seed=self.seed, batch_size=self.batch_size,
                                                                   augment=True, output_names=self.output_layer_names, float_type=self.float_type,
                                                                   input_size_factor=self.input_size_factor,
                                                                   resize_aux_label=self.resize_aux_label, resize_label=self.resize_label, output_sizes=output_sizes,
                                                                   autotune=self.autotune_dataset, prefetch=self.prefetch, num_parallel_calls=self.num_parallel_calls
                                                                   ).generate_dataset()

        print('\nPrepping validation dataset...\n')
        datasets['val'], _ = CityScapesDataset(data_dir=str(Path(self.val_dir)), label_dir=str(Path(self.val_label_dir)),
                                               seed=self.seed, batch_size=self.batch_size,
                                               augment=False, output_names=self.output_layer_names, float_type=self.float_type,
                                               input_size_factor=self.input_size_factor,
                                               resize_aux_label=self.resize_aux_label, resize_label=self.resize_label,  output_sizes=output_sizes,
                                               autotune=self.autotune_dataset, prefetch=self.prefetch, num_parallel_calls=self.num_parallel_calls
                                               ).generate_dataset()

        return datasets

    def get_model(self, mode):
        model_dict = {}
        if mode == 'train':
            print('\nCreating model...\n')
            model, loss_dict, loss_weights = generate_model(self.n_classes,
                                                            input_size_factor=self.input_size_factor,
                                                            resize_output=self.resize_output,
                                                            ds_aux=self.ds_aux, gfe_aux=self.gfe_aux, resize_aux=self.resize_aux,
                                                            summary=True)
            model_dict['loss'] = loss_dict
            model_dict['loss_weights'] = loss_weights

        elif mode == 'resume':
            print('\nLoading model...\n')
            custom_objects = {'OneHotMeanIoU': OneHotMeanIoU}
            model = keras.models.load_model(str(Path(self.save_dir, 'checkpoints')), custom_objects=custom_objects)

        else:
            raise ValueError(f"Mode {mode} not recognized.")

        if self.save_train_images:
            model.train_step = make_save_img_step(model, self.img_dir)

        output_sizes = {}
        for layer in model.layers:
            if layer.name in self.output_layer_names:
                output_sizes[layer.name] = layer.output_shape

        model_dict['model'] = model
        model_dict['output_sizes'] = output_sizes

        return model_dict

    def get_callbacks(self):
        ckpt_path = Path(self.save_dir, 'checkpoints')
        cp_cb = keras.callbacks.ModelCheckpoint(filepath=str(ckpt_path), monitor='val_loss', verbose=1, save_best_only=False)

        tb_cb = keras.callbacks.TensorBoard(log_dir=self.save_dir, histogram_freq=1, write_graph=True, write_images=False,
                                            update_freq='epoch')

        callbacks_list = [cp_cb, tb_cb]
        if self.early_stopping:
            es_cb = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50,
                                                  restore_best_weights=True)
            callbacks_list.append(es_cb)

        return callbacks_list

    def __call__(self, mode):
        callback_list = self.get_callbacks()
        model_dict = self.get_model(mode)
        model = model_dict['model']
        datasets = self.get_dataset(output_sizes=model_dict['output_sizes'])

        if mode == "train":
            initial_epoch = 0
            decay_steps = self.epochs * datasets['n_train']
            learning_rate = keras.optimizers.schedules.PolynomialDecay(self.start_learning_rate,
                                                                       decay_steps,
                                                                       self.end_learning_rate,
                                                                       power=0.9)
            optimizer = keras.optimizers.SGD(momentum=0.9, learning_rate=learning_rate)
            # metrics = [keras.metrics.CategoricalAccuracy(), OneHotMeanIoU(num_classes=self.n_classes, name='mean_iou')]
            metrics = [OneHotMeanIoU(num_classes=self.n_classes, name='mean_iou')]

            run_eagerly = self.save_train_images  # eager mode must be on if images are to be saved during training

            model.compile(loss=model_dict['loss'],
                          loss_weights=model_dict['loss_weights'],
                          optimizer=optimizer,
                          metrics=metrics,
                          run_eagerly=run_eagerly)

        else:
            initial_epoch = model.optimizer.iterations.numpy() // datasets['n_train']


        history = model.fit(datasets['train'], epochs=self.epochs, initial_epoch=initial_epoch,
                            validation_data=datasets['val'], callbacks=callback_list)
        print('\nTraining completed.\n')
        model.save(str(Path(self.save_dir, 'final_model.pb', overwrite=False)))
        print('\nModel saved.\n')
        return history


def main(args):
    tf.keras.backend.clear_session()
    tf.compat.v1.enable_eager_execution()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    config = load_config(str(Path(args.config)))
    train_config = config['train']

    # tf.debugging.experimental.enable_dump_debug_info(train_config['save_dir'], tensor_debug_mode="FULL_HEALTH",
    #                                                  circular_buffer_size=-1)

    trainer = Trainer(train_dir=train_config['train_dir'],
                      train_label_dir=train_config['train_label_dir'],
                      val_dir=train_config['val_dir'],
                      val_label_dir=train_config['val_label_dir'],
                      save_dir=train_config['save_dir'],
                      sess_name=train_config['sess_name'],
                      epochs=train_config['epochs'],
                      early_stopping=train_config['early_stopping'],
                      seed=train_config['seed'],
                      start_learning_rate=train_config['start_learning_rate'],
                      end_learning_rate=train_config['end_learning_rate'],
                      batch_size=train_config['batch_size'],
                      input_names=train_config['input_names'],
                      output_names=train_config['output_names'],
                      autotune_dataset=train_config['autotune_dataset'],
                      prefetch=train_config['prefetch'],
                      num_parallel_calls=train_config['num_parallel_calls'],
                      ds_aux=train_config['ds_aux'],
                      gfe_aux=train_config['gfe_aux'],
                      resize_aux_label=train_config['resize_aux_label'],
                      float_type=train_config['float_type'],
                      resize_label=train_config['resize_label'],
                      input_size_factor=train_config['input_size_factor'],
                      save_train_images=train_config['save_train_images']
                      )

    if args.train:
        mode = 'train'
    else:
        mode = 'resume'

    trainer(mode)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default=str(Path(__file__).parent.joinpath('config.yaml')))
    # parser.add_argument('--mode', type=str, choices=['train', 'resume'])
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--resume', action='store_true')

    args = parser.parse_args()

    if args.train and args.resume:
        raise ValueError('Can only do either --train or --resume, not both.')

    if not args.train and not args.resume:
        raise ValueError('Please use --train or --resume.')

    main(args)
