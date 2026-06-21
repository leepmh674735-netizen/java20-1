# 생성자
generator_input = layers.Input(shape=(100, ))
x = layers.Reshape((1, 1, 100))(generator_input)
x = layers.Conv2DTranspose(
    512, kernel_size=4, strides=1, padding="valid", use_bias = False
)(x)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Conv2DTranspose(
  256, kernel_size=4, strides=2, padding="same", use_bias = False
)(x)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Conv2DTranspose(
  128, kernel_size=4, strides=2, padding="same", use_bias = False
)(x)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Conv2DTranspose(
  64, kernel_size=4, strides=2, padding="same", use_bias = False
)(x)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.LeackReLU(0.2)(x)
generator_output = layers.Conv2DTranspose(
  1,
  kernel_size=4,
  strides=2,
  padding="same",
  use_bias = False,
  activation = 'tanh'
)(x)
generator = models.Model(generator_input, generator_output)

# 업샘플링 에
x = layers.UpSampling(size = 2)(x)
x = layers.Conv2D(256, kernel_size=4, strides=1, padding="same")(x)