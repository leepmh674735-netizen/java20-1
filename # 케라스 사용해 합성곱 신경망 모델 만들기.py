# 케라스 사용해 합성곱 신경망 모델 만들기
from tensorflow.keras import layers, models

input_layer = layers.Input(shape=(32, 32, 3))
conv_layer_1 = layers.Conv2D(
  filters = 10,
  kernel_size = (4,4),
  strides = 2,
  padding = 'same'
)(input_layer)
conv_layer_2 = layers.Conv2D(
      filters = 20,
      kernel_size = (3, 3),
      strides= 2,
      padding = 'same'
)(conv_layer_1)
flatten_layer = layers.Flatten()(conv_layer_2)
output_layer = layers.Dense(units=10, activation = 'softmax')(flatten_layer)
model = models.Model(input_layer, output_layer)

# 케라스의 BatchNormalization 층
from tensorflow import layers
layers.BatchNormalization(momentum = 0.9)

# 케라스의 Dropout 층
from tensorflow.keras import layers
layers.Dropout(rate = 0.25)

# 케라스를 사용해 CNN 모델 만들기
from tensorflow.keras import layers, models

input_layer = layers.Input((32, 32, 3))

x = layers.Conv2D(filters = 32, kernel_size = 3,
                  strides = 1, padding = 'same')(input_layer)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(filters = 32, kernel_size = 3, strides = 2, padding = 'same')(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(filters = 64, kernel_size = 3, strides = 2, padding = 'same')(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)

x = layers.Flatten(x)(x)

x = layers.Dense(128)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)
x = layers.Dropout(rate = 0.5)

output_layer = layers.Dense(10, activation = 'softmax')(x)

model = models.Model(input_layer, output_layer)