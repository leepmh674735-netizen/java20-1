# 인코더
encoder_input = layers.Input(
  shape=(32, 32, 1), name="encoder_input"
)
x = layers.Conv2D(32, (3, 3), strides=2, activation="relu", padding="same")(
  encoder_input
)
x = layers.Conv2D(64, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2D(128, (3, 3), strides=2, activation="relu", padding="same")(x)
shape_before_flattening = K.int_shape(x)[1:]

x = layers.Flatten()(x)
z_mean = layers.Dense(2, name="z_mean")(x)
z_log_var = layers.Dense(2, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])

encoder = models.Model(encoder_input, [z_mean, z_log_var, z], name="encoder")
  

# VAE 훈련하기
class VAE(models.Model):
  def __init___(self, encoder, decoder, **kwargs):
    super(VAE, self).__init__(**kwargs)
    self.encoder = encoder
    self.decoder = decoder
    self.total_loss_tracker =  metrics.Mean(name="total_loss")
    self.reconstruction_loss_tracker = metrics.Mean(
      name="reconstruction_loss"
    )
    self_kl_loss_tracker = metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
      return [
        self.total_loss_tracker,
        self.reconstruction_loss_tracker,
        self.kl_loss_tracker,
      ]
    
  def call(self, inputs):
    z_mean, z_log_var, z = encoder(inputs)
    reconstruction = decoder(z)
    return z_mean, z_log_var, reconstruction
  
  def train_step(self,data):
    with tf.GradientTape() as tape:
      z_mean, z_log_var, reconstruction = self(data)
      self.reconstruction_loss = tf.reduce_mean(
        500
        * losses.binary_crossentropy(
              data, self.reconstruction, axis=(1, 2, 3)
        )
      )
      kl_loss = tf.reduce_sum(
        tf.reduce_sum(
          -0.5
          * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)),
          axis = 1,
        )
      )
      total_loss = self.reconstruction_loss + kl_loss

      grads = tape.gradient(total_loss, self.train_weights)
      self.optimizer.apply__gradients(zip(grads, self.trainable_weights))

      self.total_loss_tracker.update_state(total_loss)
      self.reconstruction_loss_tracker.update_state(reconstruction_loss)
      self.k1_loss_tracker.update_state(kl_loss)
    
      return {m.name: m.result() for m in self.metrics}
    
    vae = VAE(encoder, decoder)
    vae.compile(optimizer="adam")
    vae.fit(
      train,
      epochs=5,
      batch_size=100
    )

