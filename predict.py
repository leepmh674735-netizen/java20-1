import numpy as np

# CIFAR-10 클래스 이름 정의
CLASSES = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog',
                    'frog', 'horse', 'ship', 'truck'])

# 모델을 사용하여 테스트 세트 예측하기 (x_trest 오타를 x_test로 수정)
preds = model.predict(x_test)

# 예측 확률 배열에서 가장 높은 높은 확률을 가진 인덱스를 클래스 이름으로 변환
preds_single = CLASSES[np.argmax(preds, axis=-1)]

# 실제 정답 원-핫 레이블 배열에서 1이 위치한 인덱스를 클래스 이름으로 변환
actual_single = CLASSES[np.argmax(y_test, axis=-1)]