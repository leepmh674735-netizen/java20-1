# 그레이언트 패널티 함수
def gradient_penalty(self, batch_size, real_images, fake_images):
alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
diff = fake_images = real_images
interpolated = real_images + alpha * diff

with tf.GradientTape() as gp_tape:
gp_tape.watch(interpolated)
pred = self.critic(interpolated, training=True)

grads = gp_tape.gradient(pred, [interpolated])[0]
norm = tf.sqrt(tf, reduce_sum(tf.square(grads), axis=[1, 2, 3]))
gp = tf.reduce_mean((norm - 1.0) ** 2)
return gp

