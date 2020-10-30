from pathlib import Path
import time
from argparse import ArgumentParser

import tensorflow as tf
from tensorflow import keras

from fast_scnn.utils.config import load_config
from fast_scnn.utils.dataset import CityScapesDataset
from fast_scnn.utils.labels import eval_labels
from fast_scnn.model import generate_model


class Trainer(object):
    def __init__(self, *, train_dir, train_label_dir, val_dir, val_label_dir, save_dir,
                 sess_name, autotune_dataset=False, prefetch=1, num_parallel_calls=1,
                 input_names, output_names, resize_aux, float_type,
                 epochs, early_stopping, seed=None, end_learning_rate, batch_size=12, resize_label):

        self.save_dir = Path(save_dir, sess_name)
        self.save_dir.mkdir(exist_ok=True, parents=True)

        self.input_layer_names = input_names
        self.output_layer_names = output_names

        self.epochs = epochs
        self.early_stopping = early_stopping
        self.seed = seed
        self.end_learning_rate = end_learning_rate
        self.batch_size = batch_size

        self.sess_name = time.ctime(time.time()).replace(':', '.') if sess_name is None else sess_name

        self.resize_output = False if (isinstance(resize_label, list) or isinstance(resize_label, tuple)) else True

        print('\nPrepping training dataset...\n')
        self.train_ds, self.n_train_data = CityScapesDataset(data_dir=str(Path(train_dir)), label_dir=str(Path(train_label_dir)), seed=self.seed, batch_size=self.batch_size,
                                                             augment=True, output_names=output_names, resize_aux=resize_aux,
                                                             float_type=float_type, resize_label=resize_label,
                                                             autotune=autotune_dataset, prefetch=prefetch, num_parallel_calls=num_parallel_calls
                                                             ).generate_dataset()
        print('\nPrepping validation dataset...\n')
        self.val_ds, _ = CityScapesDataset(data_dir=str(Path(val_dir)), label_dir=str(Path(val_label_dir)), seed=self.seed, batch_size=self.batch_size,
                                           augment=False, output_names=output_names, resize_aux=resize_aux,
                                           float_type=float_type, resize_label=resize_label,
                                           autotune=autotune_dataset, prefetch=prefetch, num_parallel_calls=num_parallel_calls
                                           ).generate_dataset()

    def get_model(self):
        n_classes = len(eval_labels)

        return generate_model(n_classes, resize_output=self.resize_output)

    def get_callbacks(self):
        ckpt_path = Path(self.save_dir, 'checkpoints')
        cp_cb = keras.callbacks.ModelCheckpoint(filepath=str(ckpt_path), monitor='val_loss', verbose=1, save_best_only=False)

        tb_cb = keras.callbacks.TensorBoard(log_dir=str(self.save_dir), histogram_freq=1, write_graph=True, write_images=False,
                                            update_freq='epoch')
        callbacks_list = [cp_cb, tb_cb]
        if self.early_stopping:
            es_cb = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50,
                                                  restore_best_weights=True)
            callbacks_list.append(es_cb)

        return callbacks_list

    def load_model(self):
        return keras.models.load_model(str(Path(self.save_dir, 'checkpoints')))

    def __call__(self, mode):
        callback_list = self.get_callbacks()
        # steps_per_epoch =

        if mode == 'train':
            print('\nCreating model...\n')
            model, loss_dict, loss_weights = self.get_model()
            decay_steps = self.epochs * self.n_train_data
            learning_rate = keras.optimizers.schedules.PolynomialDecay(0.045, decay_steps, self.end_learning_rate,
                                                                       power=0.5)
            optimizer = keras.optimizers.SGD(momentum=0.9, learning_rate=learning_rate)
            model.compile(loss=loss_dict, loss_weights=loss_weights, optimizer=optimizer, metrics=['accuracy'])

            print('\nModel created. Training now...\n')
            history = model.fit(self.train_ds, epochs=self.epochs, validation_data=self.val_ds, callbacks=callback_list)

        elif mode == 'resume':
            print('\nLoading model...\n')
            model = self.load_model()

            print('\nResuming training...\n')
            history = model.fit(self.train_ds, epochs=self.epochs, validation_data=self.val_ds, callbacks=callback_list)

        else:
            raise ValueError(f"Mode {mode} not recognized.")

        print('\nTraining completed.\n')
        model.save(str(Path(self.save_dir, 'final_model.pb', overwrite=False)))
        print('\nModel saved.\n')
        return history


def main(args):
    tf.keras.backend.clear_session()
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

    trainer = Trainer(train_dir=train_config['train_dir'],
                      train_label_dir=train_config['train_label_dir'],
                      val_dir=train_config['val_dir'],
                      val_label_dir=train_config['val_label_dir'],
                      save_dir=train_config['save_dir'],
                      sess_name=train_config['sess_name'],
                      epochs=train_config['epochs'],
                      early_stopping=train_config['early_stopping'],
                      seed=train_config['seed'],
                      end_learning_rate=train_config['end_learning_rate'],
                      batch_size=train_config['batch_size'],
                      input_names=train_config['input_names'],
                      output_names=train_config['output_names'],
                      autotune_dataset=train_config['autotune_dataset'],
                      prefetch=train_config['prefetch'],
                      num_parallel_calls=train_config['num_parallel_calls'],
                      resize_aux=train_config['resize_aux'],
                      float_type=train_config['float_type'],
                      resize_label=train_config['resize_label']
                      )

    trainer(args.mode)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default=str(Path(__file__).parent.joinpath('config.yaml')))
    parser.add_argument('--mode', type=str, choices=['train', 'resume'])

    args = parser.parse_args()
    main(args)
