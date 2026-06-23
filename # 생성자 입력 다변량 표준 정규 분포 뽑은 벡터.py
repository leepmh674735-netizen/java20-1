import tensorflow as tf
from tensorflow.keras import layers, models

# 1. 케라스 생성자(Generator) 정의
# 입력 크기: 100차원의 잠재 공간(Latent Space) 노이즈 벡터
generator_input = layers.Input(shape=(100,))

# 100차원 벡터를 합성곱 연산이 가능한 (1, 1, 100) 형태의 텐서로 변형
# 오타 수정: 1000 -> 100 (generator_input의 차원과 일치시킴)
x = layers.Reshape((1, 1, 100))(generator_input)

# 첫 번째 전치 합성곱 층 (1x1 -> 4x4)
x = layers.Conv2DTranspose(
    512, kernel_size=4, strides=1, padding="valid", use_bias=False
)(x)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.LeakyReLU(0.2)(x)

# 두 번째 전치 합성곱 층 (4x4 -> 8x8)
# 누락된 코드 선언부 수정 (layers.Conv2DTranspose 추가)
x = layers.Conv2DTranspose(
    256, kernel_size=4, strides=2, padding="same", use_bias=False
)(x)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.LeakyReLU(0.2)(x)

# 세 번째 전치 합성곱 층 (8x8 -> 16x16)
# 누락된 코드 선언부 수정 (layers.Conv2DTranspose 추가)
x = layers.Conv2DTranspose(
    128, kernel_size=4, strides=2, padding="same", use_bias=False
)(x)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.LeakyReLU(0.2)(x)

# 네 번째 전치 합성곱 층 및 최종 출력층 (16x16 -> 32x32)
# 이미지를 생성할 때는 보통 픽셀 범위를 -1 ~ 1로 맞추기 위해 'tanh' 활성화 함수를 사용합니다.
generator_output = layers.Conv2DTranspose(
    1,
    kernel_size=4,
    strides=2,
    padding="same",
    use_bias=False,
    activation='tanh'
)(x)

# 생성자 모델 생성
generator = models.Model(generator_input, generator_output)


# 2. 전치 합성곱 대안: 업샘플링(UpSampling2D) + 일반 합성곱(Conv2D) 예시
# (이 방식은 전치 합성곱 특유의 격자 무늬 아티팩트(Checkerboard Artifacts) 현상을 줄여줍니다)
x_up = layers.UpSampling2D(size=2)(x)
x_up = layers.Conv2D(256, kernel_size=4, strides=1, padding="same")(x_up)