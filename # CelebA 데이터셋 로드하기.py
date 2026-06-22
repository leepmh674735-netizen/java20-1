# CelebA 데이터셋 로드하기
train_data = utils.image_dataset_from_directory(
  "/app/data/celeba-dataset/img_align_celeba/img_align_celeba",
  labels=None,
  image_size=(64, 64),
  batch_size=(64, 64),
  shuffle=True,
  seed=42
  interpolation="bilinear",
)

# CelebA 데이터셋 전처리하기
def preprocess(img):
  img = tf.cast(img, "float32") / 255.0
  return img

train = train_data.map(lambda x:preprocess(x))

# 잠재 공간에서 새로운 얼굴 생성하기
grid_width, grid_height = (10, 3)
z_sample = np.random.normal(size=(grid_width * grid_height, 200))

reconstructions = decoder.predict(z_sample)

fig = plt.figure(figsize=(10, 5))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(grid_width * grid_height):
  ax = fig.add_subplot(grid_height, grid_width, i + 1)
  ax.axis(off)
  ax.imshow(reconstructions[i, :, :])


  # 로컬 이미지 파일에서 텐서플로 데이터셋 만들기
  train_data = utils.image_dataset_from_directory(
      "/app/data/lego-brick-images/dataset/",
      labels=None,
      color_mode="grayscale",
      image_size=(64, 64),
      batch_size=128,
      shuffle=True,
      seed=42,
      interpolation="bilinear",
  )

  # 레고 블록 데이터 전처리하기
  def preprocess(img):
    img = (tf.cast(img, "float32") - 127.5) /127.5
    return img
  
  train = train_data.map(lambda x: preprocess(x))