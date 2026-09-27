# Deep Learning Systems: Fashion-MNIST Classification

This project implements and evaluates a PyTorch convolutional neural network (CNN) for classifying Fashion-MNIST clothing images into ten categories. It is a Deep Learning Systems capstone submission focused on reproducible experimentation, not production deployment.

## Project deliverables

- `deep_learning.ipynb` — the executed end-to-end experiment notebook.
- `Deep_Learning_Systems_Analysis_Report.pdf` — the accompanying analysis report with citations.
- `requirements.txt` — the frozen Python environment used for the project.

## Experiment overview

Fashion-MNIST contains 28×28 grayscale images of ten clothing categories. The notebook trains two CNN configurations on a deterministic subset of the original training split:

| Configuration | Controlled change | Test accuracy |
| --- | --- | --- |
| Baseline CNN | No dropout | 84.90% |
| Experimental CNN | Dropout (`p=0.30`) after the hidden layer | 83.99% |

Both models use the same architecture, initialization seed, data split, optimizer, learning rate, batch size, and 10 training epochs. The experiment changes only dropout probability. At this training duration, the baseline slightly outperformed the dropout configuration by 0.91 percentage points.

## Model architecture

The CNN consists of two convolutional blocks followed by a fully connected classifier:

```text
Input (1 × 28 × 28)
→ Conv2d(1, 32, 3×3) + ReLU + MaxPool2d(2)
→ Conv2d(32, 64, 3×3) + ReLU + MaxPool2d(2)
→ Flatten
→ Linear(3136, 128) + ReLU
→ Dropout (experiment only)
→ Linear(128, 10 logits)
```

## Running the notebook

Activate the supplied virtual environment and launch Jupyter:

```bash
./.venv/bin/jupyter lab
```

Open `deep_learning.ipynb` and use **Run All Cells**. The notebook uses the local Fashion-MNIST files if available. If they are absent, its loading cell attempts to retrieve them through `torchvision`.

## Contents

The notebook includes dataset inspection, representative images, preprocessing, a reproducible train/validation split, training logs, accuracy and loss curves, test evaluation, per-class recall, a confusion matrix, and concrete misclassification examples. The PDF report discusses the experimental outcome, limitations, responsible-use considerations, future work, and academic references.

## Dataset

Fashion-MNIST is a public dataset introduced by Xiao, Rasul, and Vollgraf (2017). It contains 60,000 training and 10,000 test images from Zalando article images. See the notebook and report for the source citation and full references.
