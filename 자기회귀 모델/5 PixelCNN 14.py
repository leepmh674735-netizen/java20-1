import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# 1. PixelCNN 모델 네트워크 구성
inputs = layers.Input(shape=(16, 16, 1))

# 첫 번째 레이어는 자기 자신(현재 픽셀)을 보지 못하도록 Mask A를 적용합니다.
x = MaskedConv2D(mask_type="A",
                 filters=128,
                 kernel_size=7,
                 activation="relu",
                 padding="same")(inputs)
  
# 자가회귀적 특징 추출을 위한 잔차 블록(Residual Blocks) 반복
for _ in range(5):
    x = ResidualBlock(filters=128)(x)

# 이후 레이어는 이전 레이어에서 마스킹이 완료되었으므로 Mask B를 적용합니다.
for _ in range(2):
    x = MaskedConv2D(
        mask_type="B",
        filters=128,
        kernel_size=1,
        strides=1,
        activation="relu",
        padding="valid",
    )(x)

# 최종 출력 레이어 (픽셀의 클래스 확률을 예측, 여기서는 4개 카테고리 분류)
out = layers.Conv2D(
    filters=4, kernel_size=1, strides=1, activation="softmax", padding="valid"
)(x)

# 모델 생성
pixel_conn = models.Model(inputs, out)

# 2. 옵티마이저 및 모델 컴파일 (오타 수정 완료)
adam = optimizers.Adam(learning_rate=0.0005)
pixel_conn.compile(optimizer=adam, loss="sparse_categorical_crossentropy")

# 3. 모델 학습 진행
# 입력 데이터(input_data)와 타겟 데이터(output_data)의 shape가 맞는지 확인 후 실행하세요.
pixel_conn.fit(
    input_data,
    output_data,
    batch_size=128,
    epochs=150
)