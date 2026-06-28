import numpy as np
import matplotlib.pyplot as plt
from neural_network import initialize_weights, forward_pass, preprocess

data    = np.load("mnist.npz")
x_train = data["x_train"]
y_train = data["y_train"]
x_test  = data["x_test"]
y_test  = data["y_test"]

x_train, y_train, x_test, y_test = preprocess(x_train, y_train, x_test, y_test)

W1, b1, W2, b2 = initialize_weights()

print("loading your trained weights...")
try:
    W1 = np.load("W1.npy")
    b1 = np.load("b1.npy")
    W2 = np.load("W2.npy")
    b2 = np.load("b2.npy")
    print("weights loaded successfully!\n")
except FileNotFoundError:
    print("no saved weights found — run train.py first and make sure it saves weights")
    exit()

num_examples = 10
random_indices = np.random.randint(0, x_test.shape[0], num_examples)

fig, axes = plt.subplots(2, 5, figsize=(14, 6))
fig.suptitle("Model Predictions on Unseen Test Images", fontsize=14)

correct = 0

for i, idx in enumerate(random_indices):
    image  = x_test[idx:idx+1]
    label  = y_test[idx]

    predictions, _ = forward_pass(image, W1, b1, W2, b2)
    predicted_digit = np.argmax(predictions)
    confidence      = predictions[0][predicted_digit] * 100

    is_correct = predicted_digit == label
    if is_correct:
        correct += 1

    ax = axes.flat[i]
    ax.imshow(image.reshape(28, 28), cmap="gray")

    color = "green" if is_correct else "red"
    ax.set_title(
        f"predicted: {predicted_digit} ({confidence:.1f}%)\nactual: {label}",
        fontsize=9,
        color=color
    )
    ax.axis("off")

plt.tight_layout()
plt.savefig("predictions.png")
plt.show()

print(f"got {correct}/{num_examples} correct in this sample")
print()

print("=== CONFIDENCE BREAKDOWN FOR LAST IMAGE ===")
digit_names = [str(i) for i in range(10)]
probs = predictions[0] * 100
for digit, prob in sorted(zip(digit_names, probs), key=lambda x: -x[1]):
    bar = "█" * int(prob / 2)
    print(f"  {digit} : {bar} {prob:.1f}%")
