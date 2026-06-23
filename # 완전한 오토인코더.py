import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

# 1. Sampling 층 정의 (VAE용 커스텀 레이어)
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# 2. 완전한 오토인코더 모델 생성
# (주의: 이전에 정의한 encoder_input, encoder_output, decoder 모델 인스턴스가 존재해야 합니다)
autoencoder = Model(encoder_input, decoder(encoder_output))


# 3. 오토인코더 컴파일하기
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")


# 4. 오토인코더 훈련하기
# 오타 수정: validate_data -> validation_data
autoencoder.fit(
    x_train,
    x_train,
    epochs=5,
    batch_size=100,
    shuffle=True,
    validation_data=(x_test, x_test),  
)


# 5. 오토인코더를 사용하여 이미지 재구성하기
example_images = x_test[:5000]
predictions = autoencoder.predict(example_images)


# 6. 인코더를 사용하여 이미지를 임베딩하기 (2차원 잠재 공간 시각화)
embeddings = encoder.predict(example_images)

# 만약 인코더가 VAE 계열이라 출력값이 [z_mean, z_log_var, z] 형태의 리스트라면 아래 주석을 해제하고 쓰세요.
# embeddings = encoder.predict(example_images)[0] # z_mean 사용

plt.figure(figsize=(8, 8))