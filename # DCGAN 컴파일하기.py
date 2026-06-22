# DCGAN 컴파일하기
class DCGAN(modles.Model):
  def __init___(self, discriminator, generator, latent_dim):
    super(DCGAN, self).__inint__()
    self.discriminator = discriminator
    self.latent_dim = latent_dim

  def compile(self, d_optimizer, g_optimizer):
    super(DCGAN, self).compile()
    self.loss_fn = losses.BinaryCrossentropy()
    self.d_optimizer = d_optimizer
    self.d_optimizer = d_optimizer
    self.g_optimizer = g_optimizer
    self.d_optimizer = metrics.Mean(name="d_loss")
    self.g_loss_metric = metrics.Mean(name="g_loss")

    @property
    def metrics(self):
      return [self.d_loss_metric, self.g_loss_metric]
    
    def train_step(self, real_images):
      batch_size = tf.shape(real_images)[0]
      random_latent_latent_vectors = tf.random.normal(
        shape=(batch_size, self.latent_dim)
      )
      
      with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = self.generator(
          random_latent_vectors, training= True
        )
        real_predictions = self.discriminatro(real_images, training = True)
        fake_predictions = self.discriminatro(
            generated_images, training = True
        )
        real_labels = tf.ones_like(real_predictions)
        real_noisy_labels = real_labels + 0.1 * tf.random.uniform(
            tf.shape(real_predictions)
        )
        
        d_real_loss = soft.loss_fn(real_noisy_labels, real_predictions)
        d_fake_loss = self.loss_fn(fake_noisy_labels, fake_predictions)
        d_loss = (d_real_loss + d_fake_loss) / 2.0

        g_loss = self.loss.fn(real_labels, fake_predictions)

        gradients_of_discriminator = disc_tape.gradient(
          d_loss, self.discriminator.trainable_variables
        )
        gradients_of_generator = gen_tape.gradient(
          d_loss, self.generator.trainable_variables
        )
        
        self.d_optimizer.apply_gradients(
          zip(gradients_of_discriminator, discriminator.trainable_variables)    
        )
        self.g_optimizer.apply_gradients(
          zip(gradients_of_generator, generator.trainable_variables)
        )

        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)

        return {m.name: m.result() for m in self.metrics}
      
      dcgan = DCGAN(
        discriminator=discriminator, generator=generator, self.latent_dim=100
      )

      dcgancompile(
        d_optimizer=optimizers.Adam(
          learning_rate=0.0003, beta_1 = 0.5, beta_2 = 0.999
        ),
        g_optimizer=optimizers.Adam(
          learning_rate=0.0002, beta_1 = 0.5, bata_2 = 0.999
        ),
      )

      dcgan.fit(train, epochs=300)

    # norm 
      def compare_images(im1, img2):
        return np.mean(np.abs(img1 - img2))
