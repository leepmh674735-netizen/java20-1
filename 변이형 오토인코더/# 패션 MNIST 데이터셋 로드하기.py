import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras import backend as K

# 1. 패션 MNIST 데이터셋 로드하기
# 의류 이미지(28x28)와 이에 해당하는 레이블 데이터를 가져옵니다.
(x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()

# 데이터 전처리 함수 정의
# 픽셀 정규화, 패딩(28x28 -> 32x32), 채널 차원 추가를 수행합니다.
def preprocess(imgs):
    imgs = imgs.astype("float32") / 255.0  # 0~1 사이로 정규화
    # 28x28 이미지를 컨볼루션 연산에 용이하도록 상하좌우에 2픽셀씩 패딩을 넣어 32x32로 만듭니다.
    imgs = np.pad(imgs, ((0, 0), (2, 2), (2, 2)), constant_values=0.0)
    imgs = np.expand_dims(imgs, -1)  # 흑백 채널 차원 추가하여 (배치, 32, 32, 1) 형태로 변환
    return imgs

x_train = preprocess(x_train)
x_test = preprocess(x_test)


# 2. 오토인코더의 인코더(Encoder) 정의
# 이미지를 차원 축소하여 잠재 공간(Latent Space)의 2차원 벡터로 압축합니다.
encoder_input = layers.Input(
    shape=(32, 32, 1), name="encoder_input"
)
x = layers.Conv2D(32, (3, 3), strides=2, activation='relu', padding="same")(
    encoder_input
)
x = layers.Conv2D(64, (3, 3), strides=2, activation='relu', padding="same")(x)
# 오타 수정: stride -> strides
x = layers.Conv2D(128, (3, 3), strides=2, activation='relu', padding="same")(x)

# Flatten 변환 전의 텐서 크기(H, W, C)를 저장하여 디코더에서 복원할 때 사용합니다.
shape_before_flattening = K.int_shape(x)[1:]

x = layers.Flatten()(x)
# 잠재 공간을 2차원으로 설정 (시각화 목적 등의 이유)
encoder_output = layers.Dense(2, name="encoder_output")(x)

encoder = models.Model(encoder_input, encoder_output)