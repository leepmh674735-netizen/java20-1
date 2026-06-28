import re
import string
import tensorflow as tf
from tensorflow.keras import layers, models, losses

# 샘플 데이터 (filtered_data가 정의되어 있어야 합니다)
filtered_data = ["Hello, world!", "TensorFlow and LSTM are great for text generation."]

# 1. 문장 부호 양옆에 공백을 추가하는 전처리 함수
def pad_punctuation(s):
    # string.punctuation에 해당하는 모든 문장부호 앞뒤로 공백 삽입
    s = re.sub(f"([{string.punctuation}])", r' \1 ', s)
    # 연속된 공백을 하나의 공백으로 축소
    s = re.sub(' +', ' ', s)
    return s

# 데이터 전처리 적용
text_data = [pad_punctuation(x) for x in filtered_data]
        
# 텐서플로 Dataset 객체 생성 (배치 크기 32, 셔플 적용)
text_ds = tf.data.Dataset.from_tensor_slices(text_data).batch(32).shuffle(1000)

# 2. 텍스트 벡터화 레이어 설정
vocab_size = 1000  # 어휘 사전의 최대 크기
sequence_length = 200

vectorize_layer = layers.TextVectorization(
    standardize='lower',
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length + 1, # 입력(X)과 정답(Y)을 쪼개기 위해 +1 해줌
)

# 데이터셋을 통해 어휘 사전 학습
vectorize_layer.adapt(text_ds)
vocab = vectorize_layer.get_vocabulary()

# 3. 훈련 데이터셋 생성 함수 (다대다 구조를 위한 X, Y 분리)
# 예: "소년이 소녀를 만났다" -> X: "소년이 소녀를", Y: "소녀를 만났다"
def prepare_input(text):
    text = tf.expand_dims(text, -1)
    tokenized_sentences = vectorize_layer(text)
    x = tokenized_sentences[:, :-1]  # 마지막 토큰 제외 (입력)
    y = tokenized_sentences[:, 1:]   # 첫 번째 토큰 제외 (정답 다음 단어 예측)
    return x, y

# 데이터셋에 전처리 함수 매핑
train_ds = text_ds.map(prepare_input)

# 4. LSTM 모델 구축 및 컴파일
inputs = layers.Input(shape=(None,), dtype="int32")
x = layers.Embedding(input_dim=vocab_size, output_dim=100)(inputs)
x = layers.LSTM(128, return_sequences=True)(x)
outputs = layers.Dense(vocab_size, activation='softmax')(x)

lstm = models.Model(inputs, outputs)

# 컴파일 및 학습
loss_fn = losses.SparseCategoricalCrossentropy()
lstm.compile(optimizer="adam", loss=loss_fn)

# 학습 시작 (현재 샘플 데이터 수가 적어 에포크당 스텝이 매우 짧습니다)
lstm.fit(train_ds, epochs=25)