import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"

import urllib.request
print("downloading MNIST...")
urllib.request.urlretrieve(url, "mnist.npz")
print("done!")

data = np.load("mnist.npz")
x_train = data["x_train"]
y_train = data["y_train"]
x_test  = data["x_test"]
y_test  = data["y_test"]

print()
print("=== WHAT OUR DATA LOOKS LIKE ===")
print(f"training images : {x_train.shape}")
print(f"training labels : {y_train.shape}")
print(f"pixel range     : {x_train.min()} to {x_train.max()}")
print(f"first 5 labels  : {y_train[:5]}")

print()
print("=== ONE IMAGE AS RAW NUMBERS ===")
print(x_train[0])

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle("Real handwritten digits from MNIST", fontsize=14)
for i, ax in enumerate(axes.flat):
    ax.imshow(x_train[i], cmap="gray")
    ax.set_title(f"Label: {y_train[i]}", fontsize=12)
    ax.axis("off")
plt.tight_layout()
plt.show()
