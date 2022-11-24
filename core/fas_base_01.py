from keras import Input
from keras.layers import Conv2D, Dropout, GlobalAveragePooling2D
from keras.layers import Dense, BatchNormalization, MaxPooling2D, concatenate, Activation, Flatten
from keras.layers.pooling import AveragePooling2D
from keras.models import Model
from keras.utils import plot_model


def block_stem(x, filter_cnv, pool_size=(2, 2), kernel_size=7, strides=1,  activation='relu', padding='same', name="Block_Stem"):
    conv_stem = Conv2D(filter_cnv, kernel_size=kernel_size, strides=strides, activation=activation, padding=padding)(x)
    conv_stem = Conv2D(filter_cnv, kernel_size=kernel_size, strides=strides * 2, activation=activation, padding=padding)(conv_stem)
    conv_stem = MaxPooling2D(pool_size=pool_size)(conv_stem)
    conv_stem = Conv2D(filter_cnv * 2, kernel_size=kernel_size//2, strides=strides, activation=activation, padding=padding)(conv_stem)
    conv_stem = Conv2D(filter_cnv * 2, kernel_size=kernel_size//2, strides=strides * 2, activation=activation, padding=padding, name=name)(conv_stem)
    conv_stem = MaxPooling2D(pool_size=pool_size)(conv_stem)
    return conv_stem


def block_conv(x, filter_block, activation='relu', padding='same', name="block_conv"):
    conv_a = Conv2D(filter_block, kernel_size=1, strides=2,  activation=activation, padding=padding)(x)
    conv_a = Conv2D(filter_block//2, kernel_size=7, strides=1, activation=activation, padding=padding)(conv_a)
    conv_a = BatchNormalization()(conv_a)

    conv_b = Conv2D(filter_block, kernel_size=1, strides=1, activation=activation, padding=padding)(x)
    conv_b = Conv2D(filter_block//2, kernel_size=5, strides=2, activation=activation, padding=padding)(conv_b)
    conv_b = BatchNormalization()(conv_b)

    conv_c = Conv2D(filter_block, kernel_size=1, strides=2, activation=activation, padding=padding)(x)
    conv_c = Conv2D(filter_block//2, kernel_size=3, strides=1, activation=activation, padding=padding)(conv_c)
    conv_c = BatchNormalization()(conv_c)

    conv_d = Conv2D(filter_block//2, kernel_size=1, strides=2, activation=activation, padding=padding)(x)
    conv_d = BatchNormalization()(conv_d)

    conv_concat = concatenate([conv_a, conv_b, conv_c, conv_d])
    output_block = Activation('relu', name=name)(conv_concat)
    return output_block
    pass


def block_identity(x, filter_block, kernel_size_a=5, kernel_size_b=3, kernel_size_c=1, stride=1, activation='relu', padding='same', name="block_identity"):
    conv_a = Conv2D(filter_block, kernel_size=kernel_size_a, strides=stride, activation=activation, padding=padding)(x)
    conv_a = BatchNormalization()(conv_a)

    conv_b = Conv2D(filter_block, kernel_size=kernel_size_b, strides=stride, activation=activation, padding=padding)(x)
    conv_b = BatchNormalization()(conv_b)

    conv_c = Conv2D(filter_block, kernel_size=kernel_size_c, strides=stride, activation=activation, padding=padding)(x)
    conv_c = BatchNormalization()(conv_c)

    conv_concat = concatenate([conv_a, conv_b, conv_c, x])
    output_block = Activation('relu', name=name)(conv_concat)
    return output_block


def block_identity_group(x, filter_blocks, activation='relu', padding='same', name="block_identity_group"):
    for i in range(0, len(filter_blocks)):
        block_id = block_identity(x, filter_block=filter_blocks[i], activation=activation, padding=padding, name="block_identity_index_" + str(i))
        x = block_id
    output_block = Activation('relu', name=name)(x)
    return output_block


def created_model_fas_01(input_shape, number_class, activation):
    input_layer = Input(shape=input_shape)
    x_stem = block_stem(x=input_layer, filter_cnv=64, pool_size=(2, 2), kernel_size=5, strides=1, activation='relu', padding='same', name="Block_Stem")
    x_conv = block_conv(x=x_stem, filter_block=32, activation='relu', padding='same', name="block_conv_a")
    x_base_a = block_identity(x_conv, filter_block=64, activation='relu', padding='same', name="block_identity_index_1")
    x_base_b = block_identity(x_base_a, filter_block=128, activation='relu', padding='same', name="block_identity_index_2")
    conv_concat = concatenate([x_base_a, x_base_b, x_conv], name="concat_base")
    x = Flatten()(conv_concat)
    x = Dropout(0.5)(x)
    x = Dense(number_class, activation=activation)(x)
    return Model(inputs=input_layer, outputs=x)


# model = created_model_fas_01(input_shape=(300, 100, 3), number_class=1, activation='sigmoid')
# model.summary(show_trainable=True)

