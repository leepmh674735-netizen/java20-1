x_train[54, 12, 13, 1]
# 0.36862746

from tensorflow.keras import layers, models

models = models.Sequential([
  layers.Flatten(input_shape=(32, 32, 3)),
  layers.Dense(200, activation = 'relu'),
  layers.Dense(150, activation = 'relu'),
  layers.Dense(10, activation = 'softmax'),
])