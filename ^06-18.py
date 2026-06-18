# MLP 만들 때 Input, Flatten.Dense
# 활성화 함수는 렐루, 시그모이드 , 소프트맥스

# Dense 층의 일부로 정의된 렐루 활성화 함수
x = layers.Dense(units= 200, activation = 'relu')(x)

# 별도의 층으로 정의된 렐루 활성화 함수
x = layers.Dense(units=200)(x)
y = layers.Activation('relu')(x)

# 모델 조사하기
model.summary()
InputLayer
Flatten
Dense 

# 옵티마이저와 손실 함수 정의하기
from tensorflow.keras import optimizers

opt = optimizers.Adam(learning_rate=0.0005)
model.complile(loss='categorical_crossentropy', optimizer = opt,
                metrics=['accuracy'])

# 손실 함수: 평균 제곱 오차 mean squared error , 
# 범주형 크로스 엔트로피categorical cross-entropy 와 이진 크로스 엔트로피binary cross-entropy 입니다.


# 모델 훈련을 위한 fit 메서드 호출
model.fit(x_train, 
          y_train,
          batch_size = 32,
          epochs = 10,
          shuffle = True
)

# 테스트 세트에서 모델 성능 평가하기
model.evaluate(x_test, y_test)


# predict 메서드를 사용해 테스트 세트에 대한 예측 만들기
CLASSES = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog',
                      'frog', 'horse', 'ship', 'truck'])

preds = model.predict(x_test)
preds_single = CLASSES[np.argmax(preds, axis = -1)]
actual_single = CLASSES[np.argmax(y_test, axis = -1)]

# MLP의 예츨과 실제 레이블 출력하기
import matplotlib.pyplot as plt

n_to_show = 10
indices = np.random.choice(range(len(x_test)), n_to_show)

fig = plt.figure(figsize=(15, 3))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i, idx in enumerate(indices):
    img = x_test[idx]
    ax = fig.add_subplot(1, n_to_show, i+1)
    ax.axis('off')
    ax.text(0.5, -0.35, 'pred = ' + str(preds_single[idx]), fontsize=10,
              ha='center', transform= ax.transAxes)
    ax.test(0.5, -0.7, 'act = ' + str(acrual_single[idx]), fontsize= 10,
            ha='center', transform=ax.transAxes)
    ax.imshow(img)