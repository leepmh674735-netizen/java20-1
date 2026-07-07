import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers, models

# 1. PixelCNN 분포 정의
dist = tfp.distributions.PixelCNN(
    image_shape=(32, 32, 1),
    num_resnet=1,          # 오타 수정: num_resent -> num_resnet
    num_hierarchies=2,
    num_filters=32,
    num_logistic_mix=5,
    dropout_p=0.3,
)

# 2. 케라스 모델 입력 정의
image_input = layers.Input(shape=(32, 32, 1))

# 3. 로그 확률 계산
log_probs = dist.log_prob(image_input)

# 4. 모델 구축 (변수명 log_probs로 통일)
model = models.Model(inputs=image_input, outputs=log_probs)

# 5. 손실 함수 추가 (음의 로그 우도 - Negative Log-Likelihood)
model.add_loss(-tf.reduce_mean(log_probs))

# 모델 컴파일 (학습을 진행하려면 옵티마이저가 필요합니다)
model.compile(optimizer='adam')


# 6. PixelCNN 혼합 분포에서 샘플링하기 (10개의 이미지 생성)
# PixelCNN은 생성 모델이므로 train이 되지 않은 상태에서도 샘플링(초기화 상태)이 가능합니다.
generated_samples = dist.sample(10).numpy()

print("생성된 샘플 형태:", generated_samples.shape) 
# 출력: (10, 32, 32, 1)