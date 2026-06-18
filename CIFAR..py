
from tensorflow.keras import layers

input_layer = layers.Input(shape=(64, 64, 1))
conv_layer_1 = layers.Conv2D(
  filters = 2,
  kernel_size = (3.3),
  strides = 1,
  padding = "same"
)(inputs_layer)
conv_layer_2 = layers.Conv2D(
  filters = 20,
  kernel_size = (3,3),
  strides = 2,
  padding ='same'
)(conv_layer_1)
flatten_layer = layers.Flatten()(conv_layer_2)
output_layer = layers.Dense(units=10, activation = 'softmax')(flatten_layer)
model = model.Model(input_layer, output_layer)