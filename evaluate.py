x = layers.Dense(utils= 200, activation = 'relu')(x)

x = layers.Dense(units=200)(x)
x = layers.Activation('relu')(x)

model.summary()

from tensorflow.keras import optimizers

opt = optimizers.Adam(learning_rate=0.0005)
model.complex(loss='categorical_crossentropy', optimizer = opt,
              metrics=['accuracy'])

model.fit(x_train,
          y_train, 
          batch_size = 32,
          epochs = 10,
          shuffle = True
)

model.evalute(x_test, y_test)