import tensorflow as tf
from tensorflow.keras import layers, models

# 1. 흑백 이미지 입력에 적용한 간단한 Conv2D 층 예시
input_layer_mono = layers.Input(shape=(64, 64, 1))
conv_layer_mono = layers.Conv2D(  # 변수명 일치 (con_layer_1 -> conv_layer_mono)
    filters=2,
    kernel_size=(3, 3),
    strides=1,
    padding="same"
)(input_layer_mono)


# 2. 케라스를 사용해 합성곱 신경망(CNN) 모델 만들기
# 32x32 크기의 RGB 컬러 이미지 입력 레이어 정의
input_layer = layers.Input(shape=(32, 32, 3))

# 첫 번째 합성곱 층 (필터 10개, 커널 크기 4x4, 스트라이드 2, 패딩 same)
conv_layer_1 = layers.Conv2D(
    filters=10,
    kernel_size=(4, 4),
    strides=2,
    padding='same'
)(input_layer)

# 두 번째 합성곱 층 (필터 20개, 커널 크기 3x3, 스트라이드 2, 패딩 same) - 변수명 일치 (con_layer_2 -> conv_layer_2)
conv_layer_2 = layers.Conv2D(
    filters=20,
    kernel_size=(3, 3),
    strides=2,
    padding='same'
)(conv_layer_1)

# 평탄화 층 (1차원 벡터로 변환)
flatten_layer = layers.Flatten()(conv_layer_2)

# 출력 레이어 (클래스 10개, Softmax 활성화 함수) - 오타 수정 (floatten_layer -> flatten_layer)
output_layer = layers.Dense(units=10, activation='softmax')(flatten_layer)

# 함수형 API 모델 생성
model = models.Model(inputs=input_layer, outputs=output_layer)


# 3. 케라스의 BatchNormalization(배치 정규화) 층 개별 선언 예시
# (학습 과정을 안정화시키고 속도를 가속화하기 위해 사용)
bn_layer = layers.BatchNormalization(momentum=0.9)

# 4. 케라스의 Dropout(드롭아웃) 층 개별 선언 예시
# (과적합을 방지하기 위해 학습 시 무작위로 25%의 뉴런을 비활성화)
dropout_layer = layers.Dropout(rate=0.25)