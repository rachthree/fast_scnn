# Fast-SCNN

Fast-SCNN is a convolutional neural network for real-time semantic segmentation. I believe it's a very impressive architecture worth implementing as it could be applicable to autonomous vehicles. The authors did not publish their code, I'm guessing due to understandably proprietary reasons. Much kudos to them for creating this.

Some notes on my implementation:
* Auxiliary losses at the end of learning to downsample and global feature extraction modules as described in the paper.
  * Assumption: these used a similar classifier block.
* More accurate implementation of the Pyramid Pooling Module as described in Pyramid Parsing Scene Network
  * Assumption: Following what PSP did with their input residual connection, I assume 256 channels as output from the PPM as a check. The paper lists 128 channels, but could this be a typo?
* Current implementation does UpSampling2D which requires the bins to be a different size from what PSP uses
  * Resizing layer was also used, which then can match PSP for the PPM but UpSampling2D is used for now for debugging and ensuring gradients backpropagate.
* More accurate placement of batch normalizations and activations
* Conv2D regularization. The authors mentioned this was however not necessary for depthwise convolutions.
* Data augmentation as described in the paper. This implements random resizing, translation/crop, horizontal flip, color noise, and brightness.
* Uses Cityscapes as a TF Dataset
* Training driver (`train_model.py`)

## Instructions
* Install using `python setup.py develop`
* Edit `fast_scnn\config.yaml` to your needs
* Use `python train_model.py --train` for a new session or `--resume` to resume training. `--config=<path>` will specify the config to use. By default it will use the `config.yaml` in the same directory.

## Config
`config.yaml` has a number of options using the fields below:

`train`:
*  `train_dir`: (str) training images directory.
*  `train_label_dir`: (str) training labels directory.
*  `val_dir`: (str) validation images directory.
*  `val_label_dir`: (str) validation labels directory.
*  `save_dir`: (str) directory to save model.
*  `sess_name`: (str) session name for saving model.
*  `epochs`: (str) number of epochs, default 1000.
*  `early_stopping`: (Boolean) true to use early stopping, default false.
*  `seed`: (int) random seed, default None.
*  `start_learning_rate`: (float) starting learning rate, default 0.045.
*  `end_learning_rate`: (float) end learning rate, default 0.00001.
*  `batch_size`: (int) batch size, default 12.
*  `input_names`: (list of str) name of input layer, default `["input_layer"]`. **Would not recommend changing.**
*  `output_names`: (list of str) name of output layers, default `['output', 'ds_aux', 'gfe_aux']`. **Would not recommend changing.**
*  `autotune_dataset`: (Boolean) true to use TF Dataset autotuning, default false. **Would not recommend changing unless you have a lot of resources.**
*  `prefetch`: (int) number of batches to prepare before each iteration, default 1.
*  `num_parallel_calls`: (int) number of parallel calls for asynchronous dataset processing, default 1. **Would not recommend changing unless you have a lot of resources.**
*  `ds_aux`: (Boolean) true to train with the auxiliary layer at the end of learning to downsample module, default true.
*  `gfe_aux`: (Boolean) true to train with the auxiliary layer at the end of learning to global feature extractor module, default true.
*  `resize_label`: (Boolean) true to resize the final output to the output layer size sans resizing layer, default false to keep original label size and resizing layer.
*  `resize_aux_label` (dict) parent field for `ds_aux: boolean` and `gfe_aux: boolean`. If true for the auxiliary layer, there will be no final resizing for the layer and the label wil be resized to the auxiliary layer output size for training. Default: None for parent field (false for the aux keys) to keep the original label size for the auxiliary layer.
*  `float_type`: (str) float type, default "float32". "float16" also supported
*  `input_size_factor`: (float) factor to adjust the input data size, e.g. for half-resolution training, use 0.5. Default: 1.0.
*  `save_train_images`: (Boolean) true to save input and output images while training, default false. **Would not recommend changing unless debugging and you have a lot of resources. This slows training down considerably.** 

## To do:
* Finish debugging model training
* Train model to completion
* Complete evaluation code
* Conversion to fully convolutional to accept any size input
* Adding other datasets

## Sources
* Paper: https://arxiv.org/abs/1902.04502
* Dataset: https://www.cityscapes-dataset.com/

Feel free to submit issues! I will do my best to address them. I definitely want this model available for public use.
