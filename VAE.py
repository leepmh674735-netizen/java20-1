import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import utils

# CelebA 데이터셋 로드하기

train_data = utils.image_dataset_from_directory(
    "/app/data/celeba-dataset/img_align_celeba/img_align_celeba",
    labels=None,
    color_mode="rgb",
    image_size=(64, 64),
    batch_size=128,
    shuffle=True,
    seed=42,
    interpolation="bilinear",
)

# CelebA 데이터셋 전처리하기 (0~255 사이의 픽셀 값을 0.0~1.0 사이로 정규화)
def preprocess(img):
    img = tf.cast(img, "float32") / 255.0
    return img

# map 메서드를 사용하여 전체 데이터셋에 전처리 함수 적용
train = train_data.map(lambda x: preprocess(x))

# 잠재 공간(Latent Space)에서 새로운 얼굴 생성하기
grid_width, grid_height = (10, 3)

# 변수 계산 오타 수정 (grid_width * grid_width -> grid_width * grid_height)
# 잠재 공간의 차원이 200차원(VAE 표준정규분포)인 무작위 벡터 생성
z_sample = np.random.normal(size=(grid_width * grid_height, 200))

# 디코더 모델을 사용하여 생성된 벡터로부터 이미지 복원(생성)
reconstructions = decoder.predict(z_sample)

# 생성된 얼굴 이미지 시각화하기
fig = plt.figure(figsize=(18, 5))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

# 변수명 오타 수정 (grid_weight -> grid_width)
for i in range(grid_width * grid_height):
    ax = fig.add_subplot(grid_height, grid_width, i + 1)
    ax.axis("off")
    ax.imshow(reconstructions[i, :, :, :]) # RGB 이미지 출력을 위해 슬라이싱 구조 수정

plt.show()

# [잠재 공간 산술 연산 및 보간 공식 주석]
# 1. 특성 조작(Feature Manipulation): 기존 잠재 벡터(z)에 특정 속성(웃음, 안경 등) 벡터를 더해 변화를 줌
# z_new = z + alpha * feature_vector

# 2. 선형 보간(Linear Interpolation): 이미지 A의 잠재 벡터(z_A)와 이미지 B의 잠재 벡터(z_B) 사이를 부드럽게 연결 (alpa -> alpha 오타 수정)
# z_new = z_A * (1 - alpha) + z_B * alpha