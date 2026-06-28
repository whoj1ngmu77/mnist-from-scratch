# Neural Network from Scratch
> Handwritten digit classifier built using **only Python and NumPy** — no PyTorch, no TensorFlow, no ML frameworks.

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | **97.95%** |
| Train Accuracy | 99.6% |
| Parameters Trained | 101,770 |
| Training Epochs | 20 |
| Live Predictions | 10/10 correct |

---

## Architecture

```
Input Layer        Hidden Layer       Output Layer
784 neurons   →   128 neurons    →   10 neurons
(28x28 pixels)    (ReLU)             (Softmax)
```

---

## Training Progress

Loss dropped from **0.3747 → 0.0241** over 20 epochs.  
Accuracy climbed from **94.1% → 99.6%** on training data.

<img width="1568" height="561" alt="image" src="https://github.com/user-attachments/assets/474d5e8d-7ec9-467c-9106-7b6d20f13234" />

---

## Live Predictions on Unseen Test Images

All 10 predictions correct. Confidence scores above 99% on most digits.

<img width="1568" height="697" alt="image" src="https://github.com/user-attachments/assets/25535a99-4f3e-4298-ade9-197d0c1aba6e" />


---

## What's Built from Scratch

- Weight initialization (He initialization)
- Forward pass with matrix multiplication
- ReLU activation function
- Softmax output layer
- Cross-entropy loss
- Backpropagation (manual chain rule derivation)
- Mini-batch gradient descent
- Model saving and loading

---

## Project Files

| File | Purpose |
|------|---------|
| `explore_data.py` | Load and visualize MNIST dataset |
| `neural_network.py` | All math — forward pass, backward pass, activations, loss |
| `train.py` | Full training loop, epoch logging, plots, saves weights |
| `predict.py` | Load saved weights, predict on 10 random test images |

---

## Run It Yourself

```bash
pip install numpy matplotlib
python train.py
python predict.py
```

---

## Tech Stack

- **Language:** Python 3
- **Libraries:** NumPy, Matplotlib only
- **Dataset:** MNIST (60,000 train / 10,000 test images)
- **Hardware:** CPU only
