
from tensorflow.keras import layers, models

# Sequential 모델 정의
seq_model = models.Sequential([
    layers.Flatten(input_shape=(32, 32, 3)),
    layers.Dense(200, activation='relu'),
    layers.Dense(150, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 모델 구조 확인
seq_model.summary()


from tensorflow.keras import layers, models

# 1. 입력 레이어 정의 (CIFAR-10 등의 32x32x3 이미지)
input_layer = layers.Input(shape=(32, 32, 3))

# 2. 이미지 평탄화 (Flatten)
x = layers.Flatten()(input_layer)n

# 3. 첫 번째 은닉층 (노드 200개, ReLU)
x = layers.Dense(units=200, activation='relu')(x)

# 4. 두 번째 은닉층 (노드 150개, ReLU)
x = layers.Dense(units=150, activation='relu')(x)

# 5. 출력 레이어 (클래스 10개, Softmax)
output_layer = layers.Dense(units=10, activation='softmax')(x)

# 6. 모델 생성
model = models.Model(inputs=input_layer, outputs=output_layer)

# 모델 구조 확인
model.summary()