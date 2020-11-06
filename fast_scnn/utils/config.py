import yaml
import time
from copy import deepcopy
from pathlib import Path

TRAIN_DEFAULTS = {'train_dir': "E:/datasets/Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train",
                  'train_label_dir': "E:/datasets/Cityscapes/gtFine/train",
                  'val_dir': "E:/datasets/Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/val",
                  'val_label_dir': "E:/datasets/Cityscapes/gtFine/val",
                  'save_dir': str(Path(__file__).parent.parent.parent.joinpath('model_training')),
                  'sess_name': time.ctime(time.time()).replace(':', '.'),
                  'seed': None,
                  'epochs': 1000,
                  'early_stopping': False,
                  'batch_size': 12,
                  'prefetch': 1,
                  'num_parallel_calls': 1,
                  'autotune_dataset': False,
                  'end_learning_rate': 0.00001,
                  'input_names': ['input_layer'],
                  'output_names': ['output', 'ds_aux', 'gfe_aux'],
                  'resize_aux_label': None,
                  'float_type': 'float32',
                  'resize_label': False,
                  }

def load_config(config_filepath):
    with open(config_filepath, 'r') as f:
        config_loaded = yaml.safe_load(f)

    train_config = deepcopy(TRAIN_DEFAULTS)
    train_config.update(config_loaded['train'])

    config = {}
    config['train'] = train_config

    return config
