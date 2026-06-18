from tensorflow.keras import layers, models

Input_layer = layers.Input(shape=(32, 32, 3))
x = layers.Flatten()(Input_layer)
x = layers.Dense(units=200, activation = 'relu')(x)
x = layers.Dense(units=150, activation = 'relu')(x)
output_layer = layers.Dense(units=10, activation = 'softmax')(x)
model = models.Model(Input_layer, output_layer)