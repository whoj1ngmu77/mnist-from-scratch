import numpy as np
import matplotlib.pyplot as plt
from neural_network import (
    initialize_weights, forward_pass, backward_pass,
    update_weights, cross_entropy_loss, get_accuracy, preprocess
)

data    = np.load("mnist.npz")
x_train = data["x_train"]
y_train = data["y_train"]
x_test  = data["x_test"]
y_test  = data["y_test"]

x_train, y_train, x_test, y_test = preprocess(x_train, y_train, x_test, y_test)

print(f"x_train after preprocessing: {x_train.shape}")
print(f"pixel range now: {x_train.min()} to {x_train.max()}")
print()

LEARNING_RATE = 0.1
EPOCHS        = 20
BATCH_SIZE    = 64

W1, b1, W2, b2 = initialize_weights()

loss_history     = []
accuracy_history = []

print("starting training...\n")

for epoch in range(EPOCHS):
    indices = np.random.permutation(x_train.shape[0])
    x_shuffled = x_train[indices]
    y_shuffled = y_train[indices]

    epoch_loss = 0
    num_batches = 0

    for start in range(0, x_train.shape[0], BATCH_SIZE):
        x_batch = x_shuffled[start : start + BATCH_SIZE]
        y_batch = y_shuffled[start : start + BATCH_SIZE]

        predictions, cache = forward_pass(x_batch, W1, b1, W2, b2)

        loss = cross_entropy_loss(predictions, y_batch)
        epoch_loss += loss
        num_batches += 1

        dW1, db1, dW2, db2 = backward_pass(cache, W2, y_batch, LEARNING_RATE)

        W1, b1, W2, b2 = update_weights(W1, b1, W2, b2, dW1, db1, dW2, db2, LEARNING_RATE)

    avg_loss = epoch_loss / num_batches
    train_preds, _ = forward_pass(x_train[:5000], W1, b1, W2, b2)
    train_acc = get_accuracy(train_preds, y_train[:5000])

    loss_history.append(avg_loss)
    accuracy_history.append(train_acc)

    print(f"epoch {epoch+1:2d}/{EPOCHS} | loss: {avg_loss:.4f} | accuracy: {train_acc*100:.1f}%")

print()
print("=== FINAL TEST RESULTS ===")
test_preds, _ = forward_pass(x_test, W1, b1, W2, b2)
test_acc = get_accuracy(test_preds, y_test)
print(f"test accuracy: {test_acc*100:.2f}%")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(loss_history, color='red', linewidth=2)
ax1.set_title("Loss over time")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.grid(True)

ax2.plot([a*100 for a in accuracy_history], color='green', linewidth=2)
ax2.set_title("Accuracy over time")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy %")
ax2.grid(True)

plt.tight_layout()
plt.savefig("training_results.png")
plt.show()
print("saved training_results.png")
np.save("W1.npy", W1)
np.save("b1.npy", b1)
np.save("W2.npy", W2)
np.save("b2.npy", b2)
print("weights saved!")
