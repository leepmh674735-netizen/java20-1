from tensorflow.keras import layers, Model

# ==========================================
# 1. Critic (Discriminator) 입력층 구성을 위한 모델 예시
# ==========================================
critic_img_input = layers.Input(shape=(64, 64, 3), name="critic_img_input")
critic_lbl_input = layers.Input(shape=(64, 64, 2), name="critic_lbl_input")

# 채널 축(-1)을 기준으로 이미지(3채널)와 라벨(2채널)을 병합 -> (64, 64, 5)
critic_concat = layers.Concatenate(axis=-1)([critic_img_input, critic_lbl_input])

# (후속 레이어 예시: Conv2D 등으로 이어짐)
# c_output = layers.Conv2D(64, kernel_size=4, strides=2, padding="same")(critic_concat)
# critic_model = Model([critic_img_input, critic_lbl_input], c_output)


# ==========================================
# 2. Generator 입력층 구성을 위한 모델 예시
# ==========================================
gen_noise_input = layers.Input(shape=(32,), name="gen_noise_input")
gen_lbl_input = layers.Input(shape=(2,), name="gen_lbl_input")

# 노이즈(32)와 라벨(2) 벡터를 병합 -> (34,)
gen_concat = layers.Concatenate(axis=-1)([gen_noise_input, gen_lbl_input])

# 전치 합성곱(Conv2DTranspose) 연산을 위해 공간 차원 확장 -> (1, 1, 34)
gen_reshape = layers.Reshape((1, 1, 34))(gen_concat)

# (후속 레이어 예시: Conv2DTranspose 등으로 이미지 크기를 키움)
# g_output = layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same")(gen_reshape)
# generator_model = Model([gen_noise_input, gen_lbl_input], g_output)