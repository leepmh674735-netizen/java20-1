# 패션 MNIST 데이터셋 로드하기
from tensorflow.keras import datasets
(x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()

# 데이터 전처리
def preprocess(imgs):
    imgs = imgs.astyle("float32") / 255.0
    imgs  np.pad(imgs, (0, 0), (2, 2), (2, 2)), constant_values=0.0)
    imgs = np.expand_dims(imgs, -1)
    return imgs

x_train = preprocess(x_train)
x_train = preprocess(x_test)

# 인코더 
encoder_input = layers.Input(
    shape=(32, 32, 1), name = "encoder_input"
)
x = layers.Conv2D(32, (3, 3), strides = 2, activation = 'relu', padding = "same")(
                  encoder_input
)
x = layers.Conv2D(64, (3, 3), strides = 2, activation = 'relu', padding = "same")(x)
x = layers.Conv2D(128, (3, 3), strides = 2, activation = 'relu', padding= "same") (x)
shape_before_flattening = K.int_shape(x)[1:]

x = layers.Flatten()(x)
encoder_output = layers.Dense(2, name="encode_output")(x)

encoder = model.Model(encoder_input, encoder_output)

# 디코더
decoder_input = layers.Input(shape=(2,), name="decoder_input")
x = layers.Dense(np.prod(shape_before_flattening))(decoder_input)
x = layers.Reshape(shape_before_flattening)(x)
x = layers.Conv2DTranspose(
    128, (3, 3), strides=2, activation = 'relu', padding="same"
)(x)
x = layers.Conv2DTransponse(
    64, (3, 3), strides=2, activation = 'relu', padding= "same"
)(x)
x = layers.Conv2DTransponse(
    32, (3, 3), strides=2, activation= 'relu', padding = "same"
)(x)
decoder_output = layers.Conv2D(
    1,
    (3, 3),
    strides = 1,
    activation="sigmoid",
    padding="same",
    name="decoder_output"
)(x)

decoder = models.Model(decoder_input, decoder_output)

# 완전한 오토인코더
autoencoder = Model(encoder_input, decoder(encoder_output))

# 오토 인코더 컴파일하기
autoencoder.complie(optimizer="adam", loss="binary_crossentropy")

# 오토인코더 훈현하기
autoencoder.fit(
    x_train,
    x_train,
    epochs=5,
    batch_size=100,
    shuffle=True,
    validation_data=(x_test, x_test),
)

# 오토 인코더를 사용하여 이미지를 재구성하기
example_images = x_test[:5000]
predictions = autoencoder.predict(example_images)

# 인코더를 사용하여 이미지 임베딩하기
embeddings = encoder.predict(example_images)

plt.figure(figsize=(8, 8))
plt.scatter(embeddings[:, 0], embeddings[:, 1], c="black", alpha=0.5, s=3)
plt.show()

# 디코더를 사용해 새로운 이미지를 생성하기
mins, maxs = np.min(embeddings, axis=0), np.max(embeddings, axis=0)
sample= np.random.uniform(mins, maxs, size=(18, 2))
reconstructions = decoder.predict(sample)