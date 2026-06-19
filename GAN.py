import tensorflow as tf
from tensorflow.keras import utils

# 로컬 이미지 파일에서 텐서플로 데이터셋 만들기
train_data = utils.image_dataset_from_directory(
    "/app/data/lego-brick-images/dataset/",
    labels=None,
    color_mode="grayscale",
    image_size=(64, 64),
    batch_size=128,
    shuffle=True,  # batch_size=128 뒤에 빠진 쉼표(,) 추가 및 오타 수정
    seed=42,
    interpolation="bilinear",
)

# 레고 블록 데이터셋 전처리하기 (이미지 픽셀 값을 -1.0 ~ 1.0 사이로 정규화)
def preprocess(img):
    img = (tf.cast(img, "float32") - 127.5) / 127.5
    return img

# map 메서드를 사용해 전체 데이터셋에 전처리 함수 적용
train = train_data.map(lambda x: preprocess(x))