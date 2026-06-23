import tensorflow as tf
from tensorflow.keras import models, losses, metrics, optimizers
import numpy as np

# DCGAN 모델 클래스 정의 (Keras Model 상속)
class DCGAN(models.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(DCGAN, self).__init__()
        self.discriminator = discriminator  # 판별자 모델
        self.generator = generator          # 생성자 모델
        self.latent_dim = latent_dim        # 잠재 공간(노이즈) 벡터의 차원

    # 모델 컴파일 설정 (옵티마이저 및 손실 함수, 메트릭 초기화)
    def compile(self, d_optimizer, g_optimizer):
        super(DCGAN, self).compile()
        self.loss_fn = losses.BinaryCrossentropy()  # 이진 크로스엔트로피 손실 함수
        self.d_optimizer = d_optimizer              # 판별자 옵티마이저
        self.g_optimizer = g_optimizer              # 생성자 옵티마이저
        self.d_loss_metric = metrics.Mean(name="d_loss")  # 판별자 손실 평균 기록용
        self.g_loss_metric = metrics.Mean(name="g_loss")  # 생성자 손실 평균 기록용

    # 훈련 중 모니터링할 메트릭 프로퍼티 정의
    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]
    
    # 배치 단위 학습을 수행하는 커스텀 훈련 스텝
    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        # 생성자에 입력할 무작위 정규분포 노이즈 벡터 생성
        random_latent_vectors = tf.random.normal(
            shape=(batch_size, self.latent_dim)
        )
        
        # 그라디언트(기록) 테이프 시작
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # 1. 가짜 이미지 생성
            generated_images = self.generator(
                random_latent_vectors, training=True
            )
            # 2. 진짜 이미지에 대한 판별자의 예측값 계산
            real_predictions = self.discriminator(real_images, training=True)
            # 3. 가짜 이미지에 대한 판별자의 예측값 계산
            fake_predictions = self.discriminator(
                generated_images, training=True
            )
            
            # 진짜 이미지의 레이블 정답지 (1로 채움)
            real_labels = tf.ones_like(real_predictions)
            # 판별자 학습을 원활