import tensorflow as tf
from tensorflow.keras import layers

class ResidualBlock(layers.Layer):
    def __init__(self, filters, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        
        # 1x1 Convolution: 채널 수를 절반으로 줄여 연산량 감소 (Bottleneck)
        self.conv1 = layers.Conv2D(
            filters=filters // 2, kernel_size=1, activation="relu"
        )
        
        # 3x3 Masked Convolution (Type B): 이전 픽셀 정보를 유지하며 특징 추출
        self.pixel_conv = MaskedConv2D(
            mask_type="B",
            filters=filters // 2,
            kernel_size=3,
            activation="relu",
            padding="same",
        )
        
        # 1x1 Convolution: 원래 채널 수(filters)로 복원
        # ★ 수정: activation="relu"를 제거하여 음수 잔차값도 전달될 수 있도록 합니다.
        self.conv2 = layers.Conv2D(
            filters=filters, kernel_size=1, activation=None
        )

    def call(self, inputs):
        # 잔차 경로 (Residual Path) 연산
        x = self.conv1(inputs)
        x = self.pixel_conv(x)
        x = self.conv2(x)
        
        # 입력(Shortcut)과 잔차(Residual)의 합산
        # layers.add 대신 파이썬 내장 연산자(+)를 사용하는 것이 Keras Custom Layer에서 더 안전합니다.
        return inputs + x