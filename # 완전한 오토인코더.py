#  완전한 오토인코더
autoencoder = Model(encoder_input, decoder(encoder_output))

# 오토인코더 컴파일하기
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")

# 오토인코더 훈련하기
autoencoder.fit(
  x_train,
  x_train,
  epochs=5,
  batch_size=100,
  shuffle=True,
  validate_data=(x_test, x_test),  
)

# 오토 인코더를 사용하여 이미지 재구성하기
example_images = x_test[:5000]
predictions = autoencoder.predict(example_images)

# 인코더를 사용하여 이미지를 임베딩하기
embeddings = encoder.predict(example_images)

plt.figure(figsize=(8, 8))
plt.scatter(embeddings[:, 0], embeddings[:, 1], c="black", alpha=0.5, s=3)
plt.show()

# 디코더를 사용해 이미지 생성하기
mins.maxs = np.min(embeddings, axis=0), np.max(embeddings, axis=0)
sample = np.random.uniform(mins, maxs, size=(18, 2))
reconstructions = decoder.predict(sample)

# Sampling 층
class Sampling(layers.Layer):
  def call(self, inputs):
    z_mean, z_log_var = inputs
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon