import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# 1. 임의의 데이터 셋 생성 (예: CIFAR-10과 유사한 형태)
# 실제 환경에서는 tf.keras.datasets.cifar10.load_data() 등을 사용합니다.
x_train = np.random.rand(1000, 32, 32, 3)
y_train = np.random.randint(0, 10, size=(1000, 10)) # One-hot 인코딩된 상태 가정
x_test = np.random.rand(100, 32, 32, 3)
y_test = np.random.randint(0, 10, size=(100, 10))

# 2. MLP 모델 설계 (Functional API 방식)
inputs = layers.Input(shape=(32, 32, 3))
x = layers.Flatten()(inputs)

# Dense 층의 일부로 정의된 렐루 활성화 함수
x = layers.Dense(units=200, activation='relu')(x)

# 별도의 층으로 정의된 렐루 활성화 함수 (위의 코드와 동일한 효과를 냄)
x = layers.Dense(units=200)(x)
x = layers.Activation('relu')(x)

# 최종 출력층 (10개 클래스 분류를 위한 소프트맥스)
outputs = layers.Dense(units=10, activation='softmax')(x)

model = models.Model(inputs=inputs, outputs=outputs)

# 3. 모델 조사하기
model.summary()

# 4. 옵티마이저와 손실 함수 정의 및 컴파일
opt = optimizers.Adam(learning_rate=0.0005)
model.compile(

        loss='categorical_crossentropy', # 다중 분류를 위한 크로스 엔트로피
        optimizer=opt,
        metrics=['accuracy']
)

# 5. 모델 훈련을 위한 fit 메서드 호출
model.fit(
        x_train, 
        y_train,
        batch_size=32,
        epochs=10,
        shuffle=True
)

# 6. 테스트 세트에서 모델 성능 평가하기
model.evaluate(x_test, y_test)

# 7. predict 메서드를 사용해 테스트 세트에 대한 예측 만들기
CLASSES = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog',
                        'frog', 'horse', 'ship', 'truck'])

preds = model.predict(x_test)
preds_single = CLASSES[np.argmax(preds, axis=-1)]
actual_single = CLASSES[np.argmax(y_test, axis=-1)]

# 8. MLP의 예측과 실제 레이블 시각화 출력하기
n_to_show = 10
indices = np.random.choice(range(len(x_test)), n_to_show)

fig = plt.figure(figsize=(15, 3))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i, idx in enumerate(indices):
        img = x_test[idx]
ax = fig.add_subplot(1, n_to_show, i+1)
ax.axis('off')
    
    # 오타 수정: ax.text 사용 및 변수명 오타 수정
    ax.text(0.5, -0.35, 'pred = ' + str(preds_single[idx]), fontsize=10,
                ha='center', transform=ax.transAxes)
ax.text(0.5, -0.7, 'act = ' + str(actual_single[idx]), fontsize=10,
        ha='center', transform=ax.transAxes)
ax.imshow(img)

plt.show()