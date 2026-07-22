# Artificial Neural Networks — Week 3, Day 1
### Object-Oriented Implementation & Conceptual Guide

**Track:** Deep Learning Internship
**Topic:** Building `Activations`, `Neuron`, and `DenseNetwork` from scratch in NumPy, cross-checking them against Keras, and applying the same ideas to a real medical dataset.


## 📁 Repository Contents

| File | Type | Description |
|---|---|---|
| [`week_3ANN_Notebook.ipynb`](./week_3ANN_Notebook.ipynb) | Code notebook | The actual from-scratch implementation — run this to execute the classes and reproduce every result below. |
| [`week_3_ANN_Report.ipynb`](./week_3_ANN_Report.ipynb) | Report notebook | Conceptual write-up: theory, math, and figures explaining *why* the code is built the way it is. |
| [`ANN_Presentation.pptx`](./ANN_Presentation.pptx) | Slide deck | 21-slide walkthrough of the same material, for presenting the assignment. |

> 💡 Start with the **report notebook** for the concepts, then open the **code notebook** to see them run. The slide deck is a condensed version of both, for presenting.


## 🧠 What This Project Covers

The assignment rebuilds a neural network from first principles as a small set of cooperating Python classes, then validates that design against a production framework:

1. **Why Neural Networks?** — from hand-written rules to learned rules
2. **From Biology to Code** — mapping neurons to weighted sums and activations
3. **The Perceptron Era** — Rosenblatt's 1958 algorithm, its update rule, and the 1969 XOR limitation
4. **Anatomy of a Network** — input, hidden, and output layers as a `DenseNetwork`
5. **The Core Equations** — the two-line linear-combination + activation pattern every layer runs
6. **Choosing an Activation Function** — Sigmoid, Tanh, ReLU, Leaky ReLU, ELU, Swish, Softmax
7. **How Data Moves Forward** — a worked forward-propagation example, cross-checked against Keras
8. **Tuning Knobs (Hyperparameters)** — learning rate, batch size, epochs, depth, width, dropout
9. **Where ANNs Shine / Where They Struggle**
10. **Real-World Deployments** — medical diagnostics, vision, finance, NLP, recommendations, forecasting
11. **Applied Project** — a dropout-regularized Keras classifier on the Breast Cancer Wisconsin dataset
12. **Wrap-Up & Next Steps** — extending the classes with backpropagation and convolutional layers


## 🏗️ Class Design

| Class | Responsibility |
|---|---|
| `Activations` | Static-method library of activation functions (sigmoid, tanh, ReLU, leaky ReLU, ELU, swish, softmax) |
| `Neuron` | A single unit: weighted sum of inputs + bias, then an activation |
| `DenseLayer` | A full layer computed as one matrix operation: `Z = W·A_prev + b`, `A = f(Z)` |
| `DenseNetwork` | Chains `DenseLayer` objects — `layer_sizes=[2, 3, 1]` builds the full topology automatically |

The from-scratch `DenseNetwork` prediction was verified against an equivalent Keras model using identical weights — both produced the same output to within floating-point precision.



## 📊 Applied Project Results

**Dataset:** Breast Cancer Wisconsin (via scikit-learn) — 569 samples, 30 features, malignant vs. benign
**Split:** 426 training / 143 test samples
**Model:** Dropout-regularized feed-forward Keras network

| Metric | Score |
|---|---|
| Test accuracy | **97.9%** |
| Test loss | 0.038 |
| Precision (malignant / benign) | 0.96 / 0.99 |
| Recall (malignant / benign) | 0.98 / 0.98 |
| F1-score (malignant / benign) | 0.97 / 0.98 |

Training and validation curves showed no significant divergence, and the confusion matrix confirmed very few misclassifications in either direction — see the applied-project section of either notebook for the full breakdown.





## 📚 Key References

- Rosenblatt, F. (1958). *The Perceptron.* Psychological Review.
- Minsky, M., & Papert, S. (1969). *Perceptrons.* MIT Press.
- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). *Learning representations by back-propagating errors.* Nature.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning.* MIT Press.
- Srivastava, N. et al. (2014). *Dropout: A Simple Way to Prevent Neural Networks from Overfitting.* JMLR.

Full reference list is in Section 13 of the report notebook.


## 🔭 Next Steps

- Add a `.backward()` method to `DenseLayer` / `DenseNetwork` to implement backpropagation and gradient descent from scratch, instead of relying on Keras for training.
- Extend `DenseLayer` into a convolutional layer to move from tabular/vector inputs toward image data.
