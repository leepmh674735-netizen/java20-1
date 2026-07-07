# 적층 LSTM 만들기
text_in = layers.Input(shape = (None,))
embedding = layers.Embedding(total_words, embedding_size)(text_in)
x = layers.LSTM(n_units, return_sequences = True)(x)
x = layers.LSTM(n_units, return_sequences = True)(x)
probabilites = layers.Dense(total_words, activation = 'softmax')(x)
model = models.Model(text_in, probabilites)

# 양방향 GPU 츨 만들기
layer = layers.Bindirectional(layer.GPU(100))

# 케라스로 MaskedConvLayer 층 만들기
class MaskedConLayer(layers.Layer):
  