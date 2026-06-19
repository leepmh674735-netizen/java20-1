import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow.keras.backend as K

# 패션 MNIST 데이터셋 로드하기
(x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()

# 데이터 전처리
def preprocess(imgs):
    imgs = imgs.astype("float32") / 255.0 # astyle -> astype 오타 수정
    # np.pad 문법 오류 수정: 패딩 대상 뒤에 할당 연산자(=) 적용 및 튜플 구조 수정
    imgs = np.pad(imgs, ((0, 0), (2, 2), (2, 2)), mode="constant", constant_values=0.0)
    imgs = np.expand_dims(imgs, -1)
    return imgs

x_train = preprocess(x_train)
x_test = preprocess(x_test) # 두 번째 변수를 x_test로 올바르게 할당 수정

# 인코더 
encoder_input = layers.Input(
    shape=(32, 32, 1), name="encoder_input"
)
x = layers.Conv2D(32, (3, 3), strides=2, activation='relu', padding="same")(
    encoder_input
)
x = layers.Conv2D(64, (3, 3), strides=2, activation='relu', padding="same")(x)
x = layers.Conv2D(128, (3, 3), strides=2, activation='relu', padding="same")(x)
shape_before_flattening = K.int_shape(x)[1:]

x = layers.Flatten()(x)
encoder_output = layers.Dense(2, name="encode_output")(x)

encoder = models.Model(encoder_input, encoder_output) # model.Model -> models.Model 오타 수정

# 디코더
decoder_input = layers.Input(shape=(2,), name="decoder_input")
x = layers.Dense(np.prod(shape_before_flattening))(decoder_input)
x = layers.Reshape(shape_before_flattening)(x)
x = layers.Conv2DTranspose(
    128, (3, 3), strides=2, activation='relu', padding="same"
)(x)
x = layers.Conv2DTranspose( # Conv2DTransponse -> Conv2DTranspose 오타 수정
    64, (3, 3), strides=2, activation='relu', padding="same"
)(x)
x = layers.Conv2DTranspose( # Conv2DTransponse -> Conv2DTranspose 오타 수정
    32, (3, 3), strides=2, activation='relu', padding="same"
)(x)
decoder_output = layers.Conv2D(
    1,
    (3, 3),
    strides=1,
    activation="sigmoid",
    padding="same",
    name="decoder_output"
)(x)

decoder = models.Model(decoder_input, decoder_output)

# 완전한 오토인코더
autoencoder = models.Model(encoder_input, decoder(encoder_output)) # Model -> models.Model 오타 수정

# 오토 인코더 컴파일하기
autoencoder.compile(optimizer="adam", loss="binary_crossentropy") # complie -> compile 오타 수정

# 오토인코더 훈련하기 (훈현 -> 훈련 오타 수정)
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
sample = np.random.uniform(mins, maxs, size=(18, 2))
reconstructions = decoder.predict(sample)