#------------------------------------------------------------------------------
# Title: aqi_model.py
# Copyright: Danaher Digital
# Time: 2021
# Desc: This script build a model of one of various models architectures.
#------------------------------------------------------------------------------
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, \
    BatchNormalization, Activation, Input, Add, Dropout, UpSampling1D, \
    Concatenate, Conv1DTranspose
logging.basicConfig(level=logging.INFO) # Change INFO to DEBUG to diagnose.

#------------------------------------------------------------------------------
# build_dense
#------------------------------------------------------------------------------
def build_dense(num_in: int, num_out: int, num_channels: int) -> Sequential:
    """
    Builds a dense (fully connected) model.

    :param num_in: Number of timepoints (neurons) in the input layer.
    :param num_out: Number of timepoints (neurons) in the output layer.
    :param num_channels: Number of channels (arrays) in the input.
    :return: Returns model.
    """
    model = Sequential()
    model.add(Dense(num_in, input_shape=(num_in, num_channels), activation='relu', name='Dense1'))
    model.add(Flatten(name='Flatten'))
    model.add(Dense(num_out, kernel_initializer=tf.initializers.zeros(), name='Output'))
    return model

#------------------------------------------------------------------------------
# build_cnn_1D
#------------------------------------------------------------------------------
def build_cnn_1D(num_in: int, num_out: int, num_channels: int,
    num_filters: int=16, num_levels: int=3, is_bn: bool=False, drop: float=0,
    is_smooth: bool=True) -> Sequential:
    """
    Builds a 1-D CNN encoder-decoder model.

    :param num_in: Number of timepoints (neurons) in the input layer.
    :param num_out: Number of timepoints (neurons) in the output layer.
    :param num_channels: Number of channels (arrays) in the input.
    :param num_filters: Number of filters in the top level. This number
        will be doubled for each deeper level.
    :param num_levels: Number of levels in the network, 2-3 allowed.
    :param is_bn: Whether to use batch normalization.
    :param drop: Fraction to dropout, often 0.2, use 0 for none.
    :param is_smooth: Whether to encourage the output to be smoothly varying.
    :return: Returns model.
    """
    assert(num_levels >= 2 and num_levels <= 3), \
        f'!!! Levels must be between 2 and 3, not {num_levels} !!!'

    model = Sequential()

    # Top encoder layer:
    filt = num_filters
    model.add(Conv1D(filt, 3, input_shape=(num_in, num_channels), padding='same', activation='relu', name='Conv1a'))
    if is_bn: model.add(BatchNormalization())
    model.add(Conv1D(filt, 3, padding='same', activation='relu', name='Conv1b'))
    if is_bn: model.add(BatchNormalization())
    model.add(MaxPooling1D(2, name='Pool1'))
    if drop > 0: model.add(Dropout(drop))

    # Middle encoder layer:
    if num_levels == 3:
        filt *= 2
        model.add(Conv1D(filt, 3, padding='same', activation='relu', name='Conv2a'))
        if is_bn: model.add(BatchNormalization())
        model.add(Conv1D(filt, 3, padding='same', activation='relu', name='Conv2b'))
        if is_bn: model.add(BatchNormalization())
        model.add(MaxPooling1D(2, name='Pool2'))

    # Botton layer:
    filt *= 2
    model.add(Conv1D(filt, 3, padding='same', activation='relu', name='Conv3a'))
    if is_bn: model.add(BatchNormalization())
    model.add(Conv1D(filt, 3, padding='same', activation='relu', name='Conv3b'))
    if is_bn: model.add(BatchNormalization())
    if drop > 0: model.add(Dropout(drop))

    # Middle decoder layer:
    if num_levels == 3:
        filt /= 2
        model.add(UpSampling1D(2, name='Up2'))
        if is_bn: model.add(BatchNormalization())
        model.add(Conv1D(filt, 3, padding='same', activation='relu', name='Conv4a'))
        if is_bn: model.add(BatchNormalization())
        model.add(Conv1D(filt, 3, padding='same', activation='relu', name='Conv4b'))
        if is_bn: model.add(BatchNormalization())

    # Top decoder layer:
    filt /= 2
    model.add(UpSampling1D(2, name='Up1'))
    model.add(Conv1D(filt, 3, padding='same', activation='relu', name='Conv5a'))
    if is_bn: model.add(BatchNormalization())
    model.add(Conv1D(filt, 3, padding='same', activation='relu', name='Conv5b'))
    if is_bn: model.add(BatchNormalization())

    # Output layer:
    if is_smooth:
        # Neurons with identical initialization tend to do the same thing,
        # which will encourage smooth changes between neighboring neurons.
        init = tf.initializers.zeros()
    else:
        init = 'glorot_uniform'
    model.add(Flatten(name='Flatten'))
    model.add(Dense(num_out, kernel_initializer=init, name='Output'))
    return model

#------------------------------------------------------------------------------
# conv_block
#------------------------------------------------------------------------------
def conv_block(input, num_filters: int, base: str, level: int,
    is_bn: bool=False, is_res: bool=False):
    """
    Builds a convolutional block for a U-net.

    :param input: The input layer, which could be the output of a preceeding layer.
    :param num_filters: The number of filters in the convolution output.
    :param base: The base name of the layer.
    :param level: The level of the U-net where this block resides.
    :param is_bn: Whether to use batch normalization.
    :param is_res: Whether to use Resnet skip connections inside conv blocks.
    :return: Returns the output of the block.
    """
    if is_res: x_skip = input

    # Main path:
    x = Conv1D(num_filters, 3, padding="same", name=f'{base}-Conv{level}a')(input)
    if is_bn: x = BatchNormalization(name=f'{base}-BN{level}a')(x)
    x = Activation("relu", name=f'{base}-Act{level}a')(x)

    x = Conv1D(num_filters, 3, padding="same", name=f'{base}-Conv{level}b')(x)
    if is_bn: x = BatchNormalization(name=f'{base}-BN{level}b')(x)
    
    # Shortcut path for skip connection:
    if is_res:
        x_skip = Conv1D(num_filters, 3, padding="same", name=f'{base}-Conv{level}c')(x_skip)
        if is_bn: x_skip = BatchNormalization(name=f'{base}-BN{level}c')(x_skip)
        x = Add()([x, x_skip])

    x = Activation("relu", name=f'{base}-Act{level}b')(x)
    return x
    
#------------------------------------------------------------------------------
# encoder_block
#------------------------------------------------------------------------------
def encoder_block(input, num_filters: int, level: int, is_bn: bool=False,
    is_res: bool=False, is_thick: bool=False):
    """
    Builds an encoder block for a U-net, which lies on the left-hand side.

    :param input: The input layer, which could be the output of a preceeding layer.
    :param num_filters: The number of filters in the convolution output.
    :param level: The level of the U-net where this block resides.
    :param is_bn: Whether to use batch normalization.
    :param is_res: Whether to use Resnet skip connections inside conv blocks.
    :param is_thick: Whether to include an extra convolultional block.
    :return: Returns the output of the encoding and the block.
    """
    e = conv_block(input, num_filters, 'DownA', level, is_bn, is_res)
    if is_thick:
        e = conv_block(input, num_filters, 'DownB', level, is_bn, is_res)

    x = MaxPooling1D(2, name=f'Pool{level}')(e)
    return e, x

#------------------------------------------------------------------------------
# decoder_block
#------------------------------------------------------------------------------
def decoder_block(input, encoding, num_filters: int, level: int,
    is_bn: bool=False, is_res=False, is_cat=False, is_tran=False):
    """
    Builds a decoder block for a U-net, which lies on the right-hand side.

    :param input: The input layer, which could be the output of a preceeding layer.
    :param encoding: The filters from the encoder block of the same lavel.
        This is used only when is_cat is True.
    :param num_filters: The number of filters in the convolution output.
    :param level: The level of the U-net where this block resides.
    :param is_bn: Whether to use batch normalization.
    :param is_res: Whether to use Resnet skip connections inside conv blocks.
    :param is_cat: Whether to concatenate the encoder's filters with the
        decoder's filters, thus forming a U-net.
    :param is_tran: Whether to use transpose convolution instead of upsampling
    :return: Returns the output of the block.
    """
    if is_tran:
        x = Conv1DTranspose(num_filters, 2, strides=2, padding="same", name=f'Tran{level}')(input)
    else:
        x = UpSampling1D(2, name=f'Samp{level}')(input)

    if is_cat: x = Concatenate(name=f'Cat{level}')([x, encoding])

    x = conv_block(x, num_filters, 'Up', level, is_bn, is_res)
    return x

#------------------------------------------------------------------------------
# build_unet_1D
#------------------------------------------------------------------------------
def build_unet_1D(num_in: int, num_out: int, num_channels: int,
    num_filters: int=16, num_levels: int=3, is_cat: bool=True,
    is_res: bool=False, is_bn: bool=False, drop: float=0,
    is_thick_encoder: bool=False, is_long_bottom: bool=False,
    is_smooth: bool=True, is_tran: bool=False) -> Model:
    """
    Builds a 1-D U-Net encoder-decoder model.

    :param num_in: Number of timepoints (neurons) in the input layer.
    :param num_out: Number of timepoints (neurons) in the output layer.
    :param num_channels: Number of channels (arrays) in the input.
    :param num_filters: Number of filters in the top level. This number
        will be doubled for each deeper level.
    :param num_levels: number of levels in the network, 1-4 allowed.
    :param is_cat: Whether to concatenate the encoder's filters with the
        decoder's filters, thus forming a U-net.
    :param is_res: Whether to use Resnet skip connections inside each
        convolutional block.
    :param is_bn: Whether to use batch normalization.
    :param drop: Fraction to dropout, often 0.2, use 0 for none.
    :param is_thick_encoder: Whether the encoder side should use twice
        as many convolutional blocks in order to encode more features.
    :param is_long_bottom: Whether the bottom level of the network should
        have more convolutional blocks in order to encode more features.
    :param is_smooth: Whether to force the output to be smoothly varying.
    :param is_tran: Whether to use transpose convolution instead of upsampling
    :return: Returns model.
    """
    # Validate arguments:
    assert(num_levels >= 1 and num_levels <= 4), \
        f'!!! Levels must be between 1 and 4, not {num_levels} !!!'

    # Input layer (channels last):
    input_shape = (num_in, num_channels)
    inputs = Input(input_shape)

    # Top encoder layer:
    level = 1
    filt = num_filters
    if num_levels >= 2:
        e1, x = encoder_block(inputs, filt, level, is_bn, is_res)
        if drop > 0: x = Dropout(drop)(x)

    # Middle encoder layers:
    if num_levels >= 3:
        level += 1
        filt *= 2
        e2, x = encoder_block(x, filt, level, is_bn, is_res, is_thick_encoder)

    if num_levels >= 4:
        level += 1
        filt *= 2
        e3, x = encoder_block(x, filt, level, is_bn, is_res, is_thick_encoder)

    # Bottom layer:
    if num_levels == 1:
        x = conv_block(inputs, filt, 'CodeA', level, is_bn, is_res)
    else:
        level += 1
        filt *= 2
        x = conv_block(x, filt, 'CodeA', level, is_bn, is_res)
    if is_long_bottom:
        x = conv_block(x, filt, 'CodeB', level, is_bn, is_res) # Moved down here 11:45 am 9/27 for 30 training days.
        x = conv_block(x, filt, 'CodeC', level, is_bn, is_res)
        x = conv_block(x, filt, 'CodeD', level, is_bn, is_res)

    if drop > 0: x = Dropout(drop)(x)

    # Middle decoder layers:
    if num_levels >= 4:
        level -= 1
        filt /= 2
        x = decoder_block(x, e3, filt, level, is_bn, is_res, is_cat, is_tran)

    if num_levels >= 3:
        level -= 1
        filt /= 2
        x = decoder_block(x, e2, filt, level, is_bn, is_res, is_cat, is_tran)

    # Top decoder layer:
    if num_levels >= 2:
        level -= 1
        filt /= 2
        x = decoder_block(x, e1, filt, level, is_bn, is_res, is_cat, is_tran)

    # Output layer:
    if is_smooth:
        init = tf.initializers.zeros()
        # Alt: init = tf.initializers.RandomUniform(minval=-0.00001, maxval=0.00001, seed=None)
    else:
        init = 'glorot_uniform'
    x = Flatten(name='Flatten')(x)
    outputs = Dense(num_out, kernel_initializer=init, name='Output')(x)

    # Model:
    model = Model(inputs, outputs, name="U-Net")
    return model

#------------------------------------------------------------------------------
# build_model
#------------------------------------------------------------------------------
def build_model(network: str, num_in: int, num_out: int, num_channels: int,
    num_filters: int=16, num_levels: int=3, is_cat: bool=True,
    is_res: bool=False, is_bn: bool=False, drop: float=0,
    is_thick_encoder: bool=False, is_long_bottom: bool=False,
    is_smooth: bool=True, is_tran: bool=False) -> Model:
    """
    Builds the model given the network architecture parameters.
 
    :param network: Name of network architecture (Dense, CNN, Unet).
    :param num_in: Number of timepoints (neurons) in the input layer.
    :param num_out: Number of timepoints (neurons) in the output layer.
    :param num_channels: Number of channels (arrays) in the input.
    :param num_filters: Number of filters in the top level. This number
        will be doubled for each deeper level.
    :param num_levels: Number of levels in the network, 1-4 allowed.
    :param is_cat: Whether to concatenate the encoder's filters with the
        decoder's filters, thus forming a U-net.
    :param is_res: Whether to use Resnet skip connections inside each
        convolutional block.
    :param is_bn: Whether to use batch normalization.
    :param drop: Fraction to dropout, often 0.2, use 0 for none.
    :param is_thick_encoder: whether the encoder side should use twice
        as many convolutional blocks in order to encode more features.
    :param is_long_bottom: Whether the bottom level of the network should
        have more convolutional blocks in order to encode more features.
    :param is_smooth: Whether to force the output to be smoothly varying.
    :param is_tran: Whether to use transpose convolution instead of upsampling
     """
    if network == 'Dense':
        model = build_dense(num_in, num_out, num_channels)
    elif network == 'CNN':
        model = build_cnn_1D(num_in, num_out, num_channels, num_filters,
            num_levels, is_bn, drop, is_smooth)
    elif network == 'UNet':
        model = build_unet_1D(num_in, num_out, num_channels, num_filters,
            num_levels, is_cat, is_res, is_bn, drop, is_thick_encoder,
            is_long_bottom, is_smooth, is_tran)
    else:
        exit(f'!!! ERROR: unsupported network {network} !!!')

    return model
