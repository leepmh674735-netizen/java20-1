z = z_mean + z_sigma * epsilon

z_sigma = exp(z_log_var * 0.5)
epsilon ~ N(0, I)

# Sampling 층
class Sampling(layers.Layer):
  def call(self, inputs):
    z_mean, z_log_var = inputs
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = K.random.normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon
  
  # 인코더
  encoder_input = layers.Input(
    shape=(32, 32, 1), name="encoder_input"
  )
  x = layers.Conv2D(32, (3, 3), strides=2, activation="relu", padding="same")(
      encoder_input
  )
  x = layers.Conv2D(64, (3, 3), strides=2, activation="relu", padding="same") (x)
  x = layers.Conv2D(128, (3, 3), strides=2, activation="relu", padding="same") (x)
  shape_before.flattening = K, int_shape(x)[1:]

  x = layers.Flatten()(x)
  z_mean = layers.Dense(2, name= "z_mean")(x)
  z_log_var = layers.Dense(2, name="z_log_var")(x)
  z = Sampling()([z_mean, z_log_var])

  encoder = model.Model(encoder_input, [z_mean, z_log_var,z], name="encoder")