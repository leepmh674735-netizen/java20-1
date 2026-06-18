import matplotlib.pylot as plt

n_to_show = 10
indices = np.random.choice(range(len(x_test)), n_to_show)

fig = plt.figure(figsize=(15,3))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i, idx in enumerate(indices):
    img = x_test[idx]
    ax = fig.add_subplot(1, n_to_show, i+1)
    ax.axis('off')

    ax.test(0.5, -0.35, 'pred =' + str(preds_single[idx])), fontsize=10,
ha = 'center', transform= ax.transAxes)

ax.text(0.5, -0.7, 'act = ' + str(actual_single[idx]), fontsize=10,
        ha='center', transform=ax.transAxes)
ax.imshow(img)
              