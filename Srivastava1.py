# 합성곱 신경망 
# 합성곱 층 convolution layer

# 흑백 이미지 입력에 적용한 Conv2D 층
from tensorflow.keras import layers

input_layer = layers.Input(shape=(64, 64, 1))
con_layer_1 = layers.Conv2D(
  filters = 2,
  kernel_size = (3, 3),
  strides = 1,
  padding = "same"
)(input_layer)

# 케라스를 사용해 합성곱 신경만 모델 만들기
from tensorflow.keras import layers, models

input_layer = layers.Input(shape=(32, 32, 3))
conv_layer_1 = layers.Conv2D(
  filters = 10,
  kernel_size = (4, 4),
  strides = 2,
  padding = 'same'
)(input_layer)
con_layer_2 = layers.Conv2D(
  filters = 20,
  kernel_size = (3, 3),
  strides = 2,
  padding = 'same'
)(conv_layer_1)
flatten_layer = layers.Flatten()(con_layer_2)
output_layer = layers.Dense(units=10, activation = 'softmax')(floatten_layer)
model = models.Model(input_layer, output_layer)

# 케라스의 BatchNormalization층
from tensorflow.keras import layers
layers.BatchNormalization(momentum = 0.9)

# 케라스의 Droput 층
from tensorflow.keras import layers
layers.Dropout(rate = 0.25)