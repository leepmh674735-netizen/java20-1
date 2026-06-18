kl_loss = -0.5 * sum(1 + z_log_var - z_mean ^ 2 - exp(z_log_var))

# VAE 훈련하기
class VAE(models.Model):
  def __ init__(self, encoder, decoder, **kwargs):
  super(VAE, self).__init__(**kwargs)
  self.encoder = encoder
  self.decoder = decoder
  self.total_loss_tracker = metrics.Mean(name="total_loss")
  self.recontstruction_loss_tracker = metrics.Mean(
      name = "reconstruction_loss"
  )
  self_kl_loss_tracker = metrics.Mean(name="kl_loss")

  @property
  def metrics(self):
    return [
      self.total_loss_tracker,
      self.recostruction_loss_tracker,
      self.self_kl_loss_tracker,
    ]


  def call(self, inputs):
    z_mean, z_log_var, z = encoder(inputs)
    reconstruction = decoder(z)
    return z_mean, z_log_var, reconstruction
  
  def train_step(self, data):
    with tf.GradientTape() as type:
      z_mean, z_log_var, reconstruction = self(data)
      reconstruction_loss = tf.reduce_mean(
        500
        * losses.binary_crossentropy(
            data, reconstruction, axis(1, 2, 3)
        )
      )
      kl_loss = tf.reduce_mean(
        tf.reduce_sum(
          -0.5
          * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)),
          axis = 1,
        )
      )
      total_loss = reconstruction_loss + kl_loss
      grads = tape.gradient(total_loss, self.trainable_weights)
      self.optimizer.apply-gradients(zip(grads, self, trainable_weights))
      
      self.total_loss_tracker.updata_state(total_loss)
      self.reconstruction_loss_tracker.update_state(reconstruction_loss)
      self.self_kl_loss_tracker.update_state(kl_loss)

      return {m.name: m, result() for m in self.metrics}
        
    vae = VAE(encoder , decoder)
    vae.compile(optimizer="adam")
    vae.fit(
      train,
      epochs=5,
      batch_size=100x = layers.Conv2D(64, (3, 3), strides=2, activation="relu", padding="same") (x)
    )