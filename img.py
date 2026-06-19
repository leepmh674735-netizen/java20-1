import matplotlib.pyplot as plt
import numpy as np

# 테스트 세트에서 무작위로 10개의 이미지 선택하기
n_to_show = 10
indices = np.random.choice(range(len(x_test)), n_to_show)

fig = plt.figure(figsize=(15, 3))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i, idx in enumerate(indices):
    img = x_test[idx]
    ax = fig.add_subplot(1, n_to_show, i + 1)
    ax.axis('off')

    # 예측 레이블 출력 (오타 ax.test -> ax.text로 수정, 괄호 위치 올바르게 정렬)
    ax.text(
        0.5,
        -0.35,
        'pred = ' + str(preds_single[idx]),
        fontsize=10,
        ha='center',
        transform=ax.transAxes,
    )

    # 실제 레이블 출력 (들여쓰기 위치를 for 루프 안으로 올바르게 수정)
    ax.text(
        0.5,
        -0.7,
        'act = ' + str(actual_single[idx]),
        fontsize=10,
        ha='center',
        transform=ax.transAxes,
    )
    
    # 이미지 화면에 표시
    ax.imshow(img)

plt.show()