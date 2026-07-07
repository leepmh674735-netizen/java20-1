import tensorflow as tf
from tensorflow.keras import layers, models

total_words = 10000  # 예시 단어 사전 크기
embedding_size = 128 # 예시 임베딩 차원
n_units = 100        # LSTM 유닛 수

# 1. 입력 레이어
text_in = layers.Input(shape=(None,))

# 2. 임베딩 레이어
x = layers.Embedding(total_words, embedding_size)(text_in)

# 3. 첫 번째 LSTM 레이어 (기존 단방향 유지 혹은 양방향으로 변경 가능)
x = layers.Bidirectional(layers.LSTM(n_units, return_sequences=True))(x)

# 4. 두 번째 LSTM 레이어 (적층 & 양방향 적용)
# 순방향(Forward)과 역방향(Backward)의 정보를 모두 활용해 가며 층을 쌓습니다.
x = layers.Bidirectional(layers.LSTM(n_units, return_sequences=True))(x)

# 5. 출력 레이어 (각 시점별 다음 단어 예측)
probabilities = layers.Dense(total_words, activation='softmax')(x)

# 6. 모델 빌드
model = models.Model(text_in, probabilities)

# 모델 구조 요약 출력 (각 레이어의 출력 Shape와 파라미터 확인 가능)
model.summary()