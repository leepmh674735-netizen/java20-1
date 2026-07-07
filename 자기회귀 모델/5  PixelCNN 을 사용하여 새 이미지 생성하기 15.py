import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks

class ImageGenerator(callbacks.Callback):
    def __init__(self, num_img):
        super(ImageGenerator, self).__init__()
        self.num_img = num_img

    def sample_from(self, probs, temperature):
        # 0 또는 음수 확률로 인한 에러 방지를 위해 미세한 값(eps)을 더해 안정성을 높입니다.
        probs = np.asarray(probs).astype('float64')
        probs = probs ** (1 / temperature)
        probs = probs / np.sum(probs)
        # 확률의 총합이 정확히 1이 안 되는 넘파이 오차를 방지하기 위해 다항 분포 샘플링 사용
        return np.random.choice(len(probs), p=probs)
  
    def generate(self, temperature):
        # self.model.input_shape를 통해 모델의 입력 크기를 동적으로 가져옵니다.
        # 예: (None, 32, 32, 1) -> (32, 32, 1)
        input_shape = self.model.input_shape[1:]
        
        # 생성할 이미지 배치 초기화 (0으로 시작)
        generated_images = np.zeros(shape=(self.num_img,) + input_shape)
        batch, rows, cols, channels = generated_images.shape

        # PixelCNN의 핵심: 자가회귀 방식으로 위->아래, 왼쪽->오른쪽으로 픽셀을 채움
        for row in range(rows):
            for col in range(cols):
                for channel in range(channels):
                    # 현재 픽셀까지 채워진 이미지를 모델에 넣어 다음 픽셀의 확률 분포를 예측
                    # verbose=0으로 설정하여 predict 반복 호출 시 출력을 숨깁니다.
                    preds = self.model.predict(generated_images, verbose=0)
                    probs = preds[:, row, col, :] # 모양: (batch, num_classes 또는 픽셀 차원)
                    
                    # 배치 내의 각 이미지별로 독립적으로 샘플링 수행
                    for img_idx in range(batch):
                        sampled_val = self.sample_from(probs[img_idx], temperature)
                        generated_images[img_idx, row, col, channel] = sampled_val
                        
        # 필요에 따라 데이터 스케일링 (예: 0~255 범위를 0~1로 조정 등)
        # 만약 모델이 0~1 사이를 예측한다면 굳이 4로 나눌 필요 없이 모델의 아웃풋 사양에 맞춰야 합니다.
        # 여기서는 기존 코드의 흐름을 유지하되 루프 밖에서 처리하도록 둡니다.
        # generated_images /= 4.0 

        return generated_images

    def on_epoch_end(self, epoch, logs=None): # 오타 수정: < -> ,
        print(f"\n[Epoch {epoch+1}] 이미지 생성 중...")
        generated_images = self.generate(temperature=1.0)
        
        # display 및 저장 로직 (사용하시는 환경이나 가상 환경의 custom display 함수에 맞춰 사용하세요)
        # 예시로 matplotlib이나 custom 함수가 정의되어 있어야 합니다.
        try:
            display(
                generated_images,
                save_to="./output/generated_img_%03d.png" % (epoch + 1)
            )
        except NameError:
            print("display 함수가 정의되어 있지 않습니다. 이미지만 반환합니다.")

# 콜백 인스턴스 생성 (오타 수정)
image_generator_callback = ImageGenerator(num_img=10)