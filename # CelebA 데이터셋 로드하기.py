import tensorflow as tf
from tensorflow.keras import utils
import numpy as np
import matplotlib.pyplot as plt

# 1. CelebA 데이터셋 로드하기
# image_dataset_from_directory를 사용하여 디렉토리의 얼굴 이미지들을 불러옵니다.
train_data_celeba = utils.image_dataset_from_directory(
    "/app/data/celeba-dataset/img_align_celeba/img_align_celeba",
    labels=None,            # 생성 모델용이므로 레이블은 가져오지 않습니다.
    image_size=(64, 64),    # 이미지 크기를 64x64로 조정합니다.
    batch_size=64,          # 배치 크기를 64로 설정합니다.
    shuffle=True,           # 데이터를 무작위로 섞습니다.
    seed=42,                # 재현성을 위한 시드 값 고정
    interpolation="bilinear", # 이미지 크기 변경 시 사용할 보간법
)

# CelebA 데이터셋 전처리하기
# 0~255 사이의 픽셀 값을 0.0~1.0 범위의 float32 타입으로 정규화합니다.
def preprocess_celeba(img):
    img = tf.cast(img, "float32") / 255.0
    return img

train_celeba = train_data_celeba.map(lambda x: preprocess_celeba(x))


# 2. 잠재 공간(Latent Space)에서 새로운 얼굴 생성하기
grid_width, grid_height = 10, 3  # 가로 10개, 세로 3개 총 30개의 이미지를 그릴 그리드 설정
# 생성자(또는 디코더)의 입력으로 사용할 정규분포 노이즈 벡터 생성 (차원: 200)
z_sample = np.random.normal(size=(grid_width * grid_height, 200))

# 디코더 모델을 사용하여 노이즈 벡터로부터 가짜 이미지 예측/생성
# (주의: 샘플 코드의 'decoder' 변수가 사전에 정의되어 있어야 합니다)
reconstructions = decoder.predict(z_sample)

# 생성된 이미지를 시각화하기 위해 Matplotlib subplot 생성
fig = plt.figure(figsize=(10, 5))
fig.subplots_adjust(hspace=0.4, wspace=0.4) # 이미지 간 간격 조절

for i in range(grid_width * grid_height):
    ax = fig.add_subplot(grid_height, grid_width, i + 1)
    ax.axis("off") # 이미지 테두리의 축 눈금 숨기기
    ax.imshow(reconstructions[i, :, :, :]) # 생성된 이미지 출력 (RGB 전체 채널 출력 유지)


# 3. 로컬 이미지 파일에서 레고 블록 텐서플로 데이터셋 만들기
train_data_lego = utils.image_dataset_from_directory(
    "/app/data/lego-brick-images/dataset/",
    labels=None,            # 레이블 제외
    color_mode="grayscale", # 흑백(그레이스케일) 이미지로 불러옵니다.
    image_size=(64, 64),    # 이미지 크기를 64x64로 설정합니다.
    batch_size=128,         # 배치 크기를 128로 설정합니다.
    shuffle=True,           # 무작위 셔플
    seed=42,
    interpolation="bilinear",
)

# 레고 블록 데이터 전처리하기
# 픽셀 값을 0~255 범위에서 GAN 모델 학습에 최적화된 -1.0~1.0 범위로 변환합니다.
def preprocess_lego(img):
    img = (tf.cast(img, "float32") - 127.5) / 127.5
    return img

train_lego = train_data_lego.map(lambda x: preprocess_lego(x))