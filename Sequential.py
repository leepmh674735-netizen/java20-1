import numpy as np
from tensorflow.keras import layers, models

# 인덱스가 54인 이미지의 (12, 13) 위치에 있는 픽셀의 녹색 채널(1) 값 확인 및 출력값 예시 주석
x_train[54, 12, 13, 1]
# 0.36862746

# Sequential 모델을 사용하여 MLP 만들기
# (변수명 중복으로 인한 케라스 내부 메서드 오염을 방지하기 위해 'models' 변수명을 'model'로 수정)
model = models.Sequential([
    layers.Flatten(input_shape=(32, 32, 3)),
    layers.Dense(200, activation='relu'),
    layers.Dense(150, activation='relu'),
    layers.Dense(10, activation='softmax'),
])