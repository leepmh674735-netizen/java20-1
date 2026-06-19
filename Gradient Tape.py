import tensorflow as tf
from tensorflow.keras import models, metrics, losses

# KL 다이버전스 손실 공식 수식 표현
# kl_loss = -0.5 * sum(1 + z_log_var - z_mean ^ 2 - exp(z_log_var))

# VAE 훈련하기
class VAE(models.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        # 손실값 추적을 위한 메트릭 정의
        self.total_loss_tracker = metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = metrics.Mean(name="reconstruction_loss") 
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")
                    
    @property
    def metrics(self):
        # 모델 평가 시 추적할 메트릭 리스트 반환
        return [
            
            self.total_loss_tracker,
            self.reconstruction_loss_tracker, 
            self.kl_loss_tracker, 

    def call(self, inputs):
        # 인코더와 디코더를 연결하는 forward pass 구현
        z_mean, z_log_var, z = self.encoder(inputs) 
        reconstruction = self.decoder(z) 
        return z_mean, z_log_var, reconstruction
  
    def train_step(self, data):
        # 커스텀 훈련 루프 정의
        with tf.GradientTape() as tape: 
            z_mean, z_log_var, reconstruction = self(data)
            
            # 1. 재구성 손실(Reconstruction Loss) 계산
            reconstruction_loss = tf.reduce_mean(
                500 * losses.binary_crossentropy(data, reconstruction, axis=(1, 2, 3)) 
            )
            
            # 2. KL 다이버전스 손실(KL Divergence Loss) 계산
            kl_loss = tf.reduce_mean(
                tf.reduce_sum(
                    -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)),
                    axis=1,
                )