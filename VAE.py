
# CelebA 데이터셋 로드하기

trable_data = utils.image_dataset_from_directory(
  "/app/data/celeba-dataset/img_align_celeba/img_align_celeba",
  labels =None,
  color_mode="rgb",
  image_size=(64, 64),
  batch_size=128,
  shuffle=True,
  seed=42,
  interploation="bilinear",
)

# CelebA 데이터셋 전처리하기
def preprocess(img):
  img = tf.cast(img, "float32") / 255.0
  return img

train = train_data.map(lambda x: preprocess(x))

# 잠재 공간 에서 새로운 얼굴 생성하기
grid_width, grid_height= (10,3)
z_sample = np.random.normal(size=(grid_width * grid_width, 200))

reconstructions = decoder.predict(z_sample)

fig = plt.figure(figsize=(18, 5))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(grid_weight * grid_height):
      ax = fig.add_subplot(grid_height, grid_width, i + 1)
      ax.axis("off")
      ax.imshow(reconstructions[i, :, :])


z_new = z + alpha * feature_vector
z_new = z_A * (1 - alpha) + z_B * alpa
                    