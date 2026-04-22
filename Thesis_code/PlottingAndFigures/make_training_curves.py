import matplotlib.pyplot as plt

epochs = list(range(1, 26))

train_loss = [
    0.419898, 0.219643, 0.195620, 0.148446, 0.147171,
    0.129063, 0.118884, 0.123226, 0.118652, 0.118996,
    0.112657, 0.109597, 0.116560, 0.107231, 0.105810,
    0.103232, 0.105893, 0.104057, 0.102867, 0.100154,
    0.103695, 0.100217, 0.099555, 0.098828, 0.097543
]

val_loss = [
    0.211380, 0.203799, 0.220689, 0.125321, 0.167106,
    0.107911, 0.104024, 0.102349, 0.100832, 0.099264,
    0.099291, 0.097245, 0.101354, 0.102864, 0.097448,
    0.090454, 0.098235, 0.090244, 0.088353, 0.093794,
    0.089519, 0.094222, 0.090966, 0.088331, 0.086992
]

val_dice = [
    0.914553, 0.921935, 0.882150, 0.945656, 0.904277,
    0.945268, 0.950553, 0.947520, 0.950236, 0.954457,
    0.943567, 0.945545, 0.942083, 0.955010, 0.942530,
    0.949633, 0.957579, 0.952890, 0.953630, 0.942834,
    0.956845, 0.956929, 0.946018, 0.952032, 0.954863
]

val_iou = [
    0.842690, 0.855288, 0.789802, 0.896970, 0.825522,
    0.896281, 0.905805, 0.900326, 0.905225, 0.912908,
    0.893202, 0.896741, 0.890555, 0.913927, 0.891350,
    0.904120, 0.918631, 0.910049, 0.911392, 0.891895,
    0.917276, 0.917437, 0.897591, 0.908477, 0.913641
]

fig, axes = plt.subplots(3, 1, figsize=(8, 10))

axes[0].plot(epochs, train_loss, marker='o', label='Train loss')
axes[0].plot(epochs, val_loss, marker='o', label='Validation loss')
axes[0].set_title('Loss across epochs')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs, val_dice, marker='o')
axes[1].set_title('Validation Dice across epochs')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Dice')
axes[1].grid(True, alpha=0.3)

axes[2].plot(epochs, val_iou, marker='o')
axes[2].set_title('Validation IoU across epochs')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('IoU')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()