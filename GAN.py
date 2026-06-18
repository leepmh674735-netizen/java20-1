# 로컬 이미지파일에서 텐서플로 데이터셋 만들기
train_data = utils.image_dataset_from_directory(
    "/app/data/lego-brick-images/dataset/",
    labels=None,
    color_mode="grayscale",
    image_size = (64, 64),
    batch_size = 128
    shuffle=True,
    seed=42,
    interpolation="bilinear",
)

# 레고 블록 데이터셋 전처리하기
def preprocess(img):
  img = (tf.cast(img,"float32") -127.5) /127.5
  return img

train = train_data.map(lambda x: preprocess(x))