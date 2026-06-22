# CIFAR-10 데이터셋 전처리
import numpy as np
from tensorflow.keras import datasets, utils

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

NUM_CLASSES = 10

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

y_train = utils.to_categorical(y_train, NUM_CLASSES)
y_test = utils.to_categorical(y_test, NUM_CLASSES)

# 인덱스 54인 이미지의 (12, 13) 위치에 있는 픽셀 녹색 채널 값
x_train[54, 12, 13, 1]
# 0.36862746

# Sequential 모델을 사용하여 MLP 만들기
from tensorflow.keras import layers, models

models = models.Sequential([
  layers.Flatten(input_shape=(32, 32, 3)),
  layers.Dense(200, activation = 'relu'),
  layers.Dense(150, activation = 'relu'),
  layers.Dense(10, activation = 'softmax'),
])

# 함수형 API를 사용하여 MLP 만들기
from tensorflow.keras import layers, models

input_layer = layers.Input(shape=(32, 32, 3))
x = layers.Flatten()(input_layer)
x = layers.Dense(units=200, activation = 'relu')(x)

x = layers.Dense(units=150, activation = 'relu')(x)
output_layer = layers.Dense(units=10, activation = 'softmax')(x)
model = models.Model(input_layer, output_layer)

# Dense 층의 일부로 정의된 렐루 활성화 함수
x = layers.Dense(units=200, activation = 'relu')(x)

# 별도의 층으로 정의된 렐루 활성화 함수
x = layers.Dense(units=200)(x)
x = layers.Activation('relu')(x)

# 모델의 조사하기
model.summary()

# 옵티마이저와 손실 함수 정의하기
from tensorflow.keras import optimizers

opt = optimizers.Adam(learning_rate=0.0005)
model.compile(loss='categorical_crossentropy', optimizer=opt,
              metrics=['accuracy'])


