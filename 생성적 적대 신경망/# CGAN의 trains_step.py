import tensorflow as tf

# CGAN의 train_step (메서드명 오타 수정: train_stop -> train_step)
def train_step(self, data):
    # 1. 데이터 언패킹
    real_images, one_hot_labels = data

    # 2. 라벨의 차원을 이미지 크기(64x64)에 맞게 확장 (콤마 오타 수정)
    image_one_hot_labels = one_hot_labels[:, None, None, :]
    image_one_hot_labels = tf.repeat(
        image_one_hot_labels, repeats=64, axis=1
    )
    image_one_hot_labels = tf.repeat(
        image_one_hot_labels, repeats=64, axis=2
    )

    # 변수명 오타 수정 (real_image -> real_images)
    batch_size = tf.shape(real_images)[0]

    # 3. Critic(Discriminator) 학습 루프 (세미콜론 오타 및 들여쓰기 수정)
    for i in range(self.critic_steps):
        random_latent_vectors = tf.random.normal(
            shape=(batch_size, self.latent_dim)
        )

        with tf.GradientTape() as tape:
            # 가짜 이미지 생성
            fake_images = self.generator(
                [random_latent_vectors, one_hot_labels], training=True
            )
            # Critic 예측 (가짜 및 진짜)
            fake_predictions = self.critic(
                [fake_images, image_one_hot_labels], training=True
            )
            real_predictions = self.critic(
                [real_images, image_one_hot_labels], training=True
            )

            # WGAN Loss 계산 (진짜 predictions가 앞에 오는 것이 표준 방향입니다)
            c_wass_loss = tf.reduce_mean(fake_predictions) - tf.reduce_mean(
                real_predictions
            )
            # Gradient Penalty 계산
            c_gp = self.gradient_penalty(
                batch_size, real_images, fake_images, image_one_hot_labels
            )

            c_loss = c_wass_loss + c_gp * self.gp_weight

        # Critic 가중치 업데이트 (들여쓰기 수정)
        c_gradient = tape.gradient(c_loss, self.critic.trainable_variables)
        self.c_optimizer.apply_gradients(
            zip(c_gradient, self.critic.trainable_variables)
        )

    # 4. Generator 학습
    random_latent_vectors = tf.random.normal(
        shape=(batch_size, self.latent_dim)
    )

    with tf.GradientTape() as tape:
        fake_images = self.generator(
            [random_latent_vectors, one_hot_labels], training=True
        )
        fake_predictions = self.critic(
            [fake_images, image_one_hot_labels], training=True
        )
        # Generator Loss: Critic이 가짜를 진짜(높은 점수)로 예측하도록 유도
        # (WGAN에서는 타겟 방향에 따라 -tf.reduce_mean(fake_predictions)를 주로 사용합니다)
        g_loss = -tf.reduce_mean(fake_predictions)

    # Generator 가중치 업데이트
    gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
    self.g_optimizer.apply_gradients(
        zip(gen_gradient, self.generator.trainable_variables)
    )
    
    # Keras가 loss를 트래킹할 수 있도록 딕셔너리 반환
    return {"c_loss": c_loss, "g_loss": g_loss}