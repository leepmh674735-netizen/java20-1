import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# ==========================================
# 1. 케라스로 MaskedConvLayer 층 만들기 (오류 수정본)
# ==========================================
class MaskedConvLayer(layers.Layer):
    def __init__(self, mask_type, **kwargs):
        super(MaskedConvLayer, self).__init__()
        self.mask_type = mask_type
        # 오타 수정: cov -> conv
        self.conv = layers.Conv2D(**kwargs)

    def build(self, input_shape):
        # 내부 Conv2D 레이어의 빌드를 먼저 호출하여 가중치(kernel, bias)를 생성합니다.
        self.conv.build(input_shape)
        
        # 커널의 형태를 가져옵니다. (예: [kernel_size, kernel_size, in_channels, out_channels])
        kernel_shape = self.conv.kernel.shape
        
        # 1. 모두 0으로 채워진 마스크 베이스 행렬 생성 (넘파이 활용)
        mask = np.zeros(kernel_shape, dtype=np.float32)
        
        # 2. 중심점 기준 윗부분 행들을 모두 1.0으로 채움
        mask[: kernel_shape[0] // 2, ...] = 1.0
        
        # 3. 중심점 행에서 중심점 왼쪽 열들을 모두 1.0으로 채움
        mask[kernel_shape[0] // 2, : kernel_shape[1] // 2, ...] = 1.0
        
        # 4. Mask B 타입인 경우에만 자기 자신(중심점 픽셀)의 정보를 허용 (1.0 추가)
        if self.mask_type == "B":
            mask[kernel_shape[0] // 2, kernel_shape[1] // 2, ...] = 1.0
            
        # 5. 생성한 마스크를 상수(상태 변수)로 저장하여 연산에 활용합니다. (학습되지 않는 가중치)
        self.mask = tf.constant(mask, dtype=tf.float32)

    def call(self, inputs):
        # 중요: 가중치 자체를 변형(assign)시키지 않고, 
        # 마스크가 적용된 가중치를 '계산'에만 적용하여 미분 흐름을 유지합니다.
        masked_kernel = self.conv.kernel * self.mask
        
        # 텐서플로우 로우레벨 연산 함수를 통해 마스킹된 커널로 컨볼루션 수행
        return tf.nn.conv2d(
            inputs, 
            filters=masked_kernel, 
            strides=self.conv.strides, 
            padding=self.conv.padding.upper() # 'same' 또는 'valid'를 대문자로 변환
        ) + self.conv.bias


# ==========================================
# 2. 기존 적층 LSTM 및 양방향 층 (참고용 정돈 코드)
# ==========================================
total_words = 10000
embedding_size = 128
n_units = 100

text_in = layers.Input(shape=(None,))
x = layers.Embedding(total_words, embedding_size)(text_in)
x = layers.LSTM(n_units, return_sequences=True)(x)
x = layers.LSTM(n_units, return_sequences=True)(x)
probabilities = layers.Dense(total_words, activation='softmax')(x)
model = models.Model(text_in, probabilities)

# 양방향 레이어 정의 (오타 수정 및 GPU 가속은 케라스 LSTM 내부에서 자동 처리됨)
bi_lstm_layer = layers.Bidirectional(layers.LSTM(100, return_sequences=True))
