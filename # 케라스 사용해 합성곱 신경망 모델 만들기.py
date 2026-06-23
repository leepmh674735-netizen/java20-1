import tensorflow as tf
from tensorflow.keras import layers, models

# 1. 케라스 사용해 간단한 합성곱 신경망(CNN) 모델 만들기
input_layer_1 = layers.Input(shape=(32, 32, 3))

conv_layer_1 = layers.Conv2D(
    filters=10,
    kernel_size=(4, 4),
    strides=2,
    padding='same'
)(input_layer_1)

conv_layer_2 = layers.Conv2D(
    filters=20,
    kernel_size=(3, 3),
    strides=2,
    padding='same'
)(conv_layer_1)

flatten_layer_1 = layers.Flatten()(conv_layer_2)
output_layer_1 = layers.Dense(units=10, activation='softmax')(flatten_layer_1)

model_1 = models.Model(input_layer_1, output_layer_1)


# 2. 케라스의 BatchNormalization 및 Dropout 층 예시
# 오타 수정: from tensorflow import layers -> layers는 keras 하위에 있습니다.
bn_layer_example = layers.BatchNormalization(momentum=0.9)
dropout_layer_example = layers.Dropout(rate=0.25)


# 3. 케라스를 사용해 더 깊은 구조의 CNN 모델 만들기
input_layer_2 = layers.Input(shape=(32, 32, 3))

# 첫 번째 합성곱 블록
x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(input_layer_2)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)

# 두 번째 합성곱 블록 (크기 축소)
x = layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)

# 세 번째 합성곱 블록 (크기 축소 및 채널 확장)
x = layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Le