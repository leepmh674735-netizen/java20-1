import tensorflow as tf
from tensorflow.keras import layers, models

# 케라스 판별자(Discriminator) 정의
# 입력 크기: 64x64 크기의 흑백(1채널) 이미지
discriminator_input = layers.Input(shape=(64, 64, 1)) 

# 첫 번째 컨볼루션 층 (64x64 -> 32x32)
x = layers.Conv2D(
    64, kernel_size=4, strides=2, padding="same", use_bias=False
)(discriminator_input)
x = layers.LeakyReLU(0.2)(x)
x = layers.Dropout(0.3)(x)

# 두 번째 컨볼루션 층 (32x32 -> 16x16)
x = layers.Conv2D(
    128, kernel_size=4, strides=2, padding="same", use_bias=False
)(x)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Dropout(0.3)(x)

# 세 번째 컨볼루션 층 (16x16 -> 8x8)
# 오타 수정: layer -> layers
x = layers.Conv2D(
    256, kernel_size=4, strides=2, padding="same", use_bias=False
)(x)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.LeakyReLU(0.2)(x)

# 네 번째 컨볼루션 층 (8x8 -> 4x4)
x = layers.Conv2D(
    512, kernel_size=4, strides=2, padding="same", use_bias=False
)(x)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Dropout(0.3)(x)

# 다섯 번째 컨볼루션 층 (4x4 -> 1x1)
# 오타 및 누락 수정: "Valid" -> "valid", 쉼표(,) 추가