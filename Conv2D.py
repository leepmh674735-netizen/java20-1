
from tensorflow.keras import layers

input_layer = layers.Input(shape=(64, 64, 1))

conv_layer_1 = layers.Conv2D(
    filters=2,
    kernel_size=(3, 3),
    strides=1,
    padding="same"
)(input_layer)