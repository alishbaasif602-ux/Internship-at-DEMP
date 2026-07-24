# Regularization, Hyperparameter Optimization & Training Strategies for ANNs
### Technical Report

**Deliverables in this assignment:**
- `ANN_Regularization_Hyperparameter_Optimization.ipynb` — fully executed Jupyter notebook (Tasks 1–7: code, tables, charts)
- `ANN_Regularization_Hyperparameter_Optimization.pptx` — 13-slide presentation deck summarizing this report for a live walkthrough
- `README.md` (this file) — the written technical report (Sections 1–12)

**Dataset:** Breast Cancer Wisconsin (Diagnostic) — 569 samples, 30 features, binary classification (malignant / benign)
**Implementation:** a from-scratch NumPy neural-network engine (Dense, ReLU, Sigmoid, Dropout, BatchNorm layers with manual forward/backward propagation and momentum-SGD), used in place of TensorFlow/Keras or PyTorch because this environment has no internet access to install them. Every result below is copied from an actual executed run of the companion notebook, not simulated.


## 1. Training Challenges in ANNs

Training an artificial neural network well is harder than getting it to run at all. The main obstacles are:

- **Underfitting.** The network is too simple, trained for too few epochs, or the learning rate is too low/high for it to fit even the training data well. Symptom: both training and validation loss stay high; both accuracies stay low.
- **Overfitting.** The network memorizes the training set instead of learning generalizable patterns — especially likely on small datasets (here, only 341 training examples). Symptom: training loss keeps falling toward zero while validation loss flattens and then rises; training accuracy reaches 100% while validation accuracy stalls. This is exactly what the Task 1 baseline shows below.
- **Vanishing / exploding gradients.** In deeper networks, gradients shrink or blow up as they propagate backward through many layers, stalling learning. Sensitive to activation function choice and weight initialization.
- **Poor weight initialization.** Initializing weights too small, too large, or with the wrong variance for the activation function slows or destabilizes convergence (demonstrated in Section 5 / Task 4).
- **Sensitivity to learning rate.** Too low → painfully slow convergence or getting stuck; too high → the loss oscillates or diverges outright (demonstrated in Section 8 / Task 6a, where `lr=0.5` collapses to 62% test accuracy).
- **Internal covariate shift.** As parameters update, the distribution of each layer's inputs keeps shifting, forcing later layers to continuously re-adapt — motivates Batch Normalization (Section 4).
- **Small-data instability.** With few samples, model performance can vary meaningfully with the random train/validation split and initialization seed, which is why validation curves and multiple metrics (not a single accuracy number) are used throughout this report.
- **Hyperparameter interdependence.** Learning rate, batch size, network depth/width, and regularization strength all interact — tuning one in isolation can be misleading (Section 8).


## 2. Regularization Methods

Regularization is any technique that trades a small amount of training-set fit for better generalization to unseen data. Methods explored or referenced in this project:

| Method | Mechanism | Covered |
|---|---|---|
| Dropout | Randomly zero a fraction of activations each forward pass, forcing redundancy | Section 3 |
| Batch Normalization | Normalizes layer inputs per mini-batch, indirectly regularizing and stabilizing training | Section 4 |
| Weight initialization | Not regularization per se, but strongly affects the trajectory optimization takes | Section 5 |
| Early stopping | Halts training once validation performance stops improving, capping how long the model can overfit | Section 7 |
| L1 / L2 weight decay | Penalizes large weights directly in the loss function (not run as a separate experiment here, but implemented implicitly by the momentum-SGD update; recommended as a follow-up experiment) | — |
| Data augmentation | Synthetically expands the training set (not applicable to this tabular dataset, but standard for image tasks) | — |

The experiments in this report focus on Dropout, Batch Normalization, weight initialization, early stopping, and learning-rate scheduling, since these are the techniques the assignment specifies.


## 3. Dropout

Dropout (Srivastava et al., 2014) randomly deactivates a fraction *p* of neurons on each training forward pass (inverted dropout: surviving activations are scaled by `1/(1-p)` so no rescaling is needed at inference time). This prevents neurons from co-adapting to fix each other's mistakes, effectively training an ensemble of thinned sub-networks that share weights.

**Result (150 epochs, `hidden=(16,8)`, He init):**

| Config | Final Train Acc | Final Val Acc | Final Train Loss | Final Val Loss | Train–Val Gap | Test Acc |
|---|---|---|---|---|---|---|
| No Dropout | 1.0000 | 0.9825 | 0.0023 | 0.0429 | 0.0175 | 0.9825 |
| Dropout 0.3 | 0.9971 | 0.9737 | 0.0060 | 0.2128 | 0.0234 | 0.9737 |
| Dropout 0.5 | 0.9912 | 0.9737 | 0.0113 | 0.1363 | 0.0175 | 0.9825 |

On this particular dataset the "No Dropout" baseline was already well-regularized enough (thanks to a small network and standardized inputs) that dropout did not reduce the train/validation gap — in fact validation loss got noisier at 0.3 before settling lower at 0.5. This is a useful negative result: dropout is not automatically beneficial, and its effect depends on how much the network is already overfitting and on network capacity relative to dataset size. Dropout tends to help more on larger networks / larger datasets (e.g., image classifiers) than on a compact 30-feature tabular problem.


## 4. Batch Normalization

Batch Normalization (Ioffe & Szegedy, 2015) normalizes each layer's pre-activation outputs to zero mean / unit variance per mini-batch, then applies a learnable scale (γ) and shift (β). This reduces internal covariate shift and generally allows larger learning rates and faster convergence.

**Result (150 epochs, `hidden=(16,8)`, He init):**

| Config | Final Train Acc | Final Val Acc | Final Val Loss | Test Acc | Epochs to 90% Train Acc |
|---|---|---|---|---|---|
| No BatchNorm | 1.0000 | 0.9737 | 0.0744 | 0.9825 | 0 |
| With BatchNorm | 1.0000 | 0.9561 | 0.2776 | 0.9474 | 1 |

Both configurations already reach 90% training accuracy essentially immediately (within the first epoch) on this small, standardized, low-dimensional dataset, so the usual "faster convergence" benefit of BatchNorm is not visible here — there's no slow-convergence problem for it to fix. Its validation loss was also higher in this run, suggesting the extra learnable parameters and per-batch noise from batch statistics (with `batch_size=32` on ~341 training examples, i.e. very few mini-batches per epoch) added variance without a compensating benefit. BatchNorm's advantages are much more pronounced on deeper networks, larger datasets, and unnormalized/raw inputs — none of which apply directly to this experiment.


## 5. Weight Initialization

Initialization determines the starting scale of activations and gradients before any learning happens:

- **Random (naive):** small `N(0, 0.01²)` weights — a common historical default with no principled variance target.
- **Xavier / Glorot** (Glorot & Bengio, 2010): scales variance by `2/(fan_in + fan_out)`, derived to keep activation variance stable for tanh/sigmoid-style symmetric activations.
- **He** (He et al., 2015): scales variance by `2/fan_in`, derived specifically to compensate for ReLU zeroing out half of its inputs.

**Result (150 epochs, `hidden=(16,8)`):**

| Init | Final Train Acc | Final Val Acc | Loss @ Epoch 10 | Final Train Loss | Test Acc | Epochs to 90% Train Acc |
|---|---|---|---|---|---|---|
| Random | 1.0000 | 0.9649 | 0.6598 | 0.0005 | 0.9737 | 14 |
| Xavier | 1.0000 | 0.9649 | 0.0326 | 0.0003 | 0.9825 | 0 |
| He | 1.0000 | 0.9649 | 0.0359 | 0.0017 | 0.9561 | 0 |

The naive random initialization is the clear laggard: at epoch 10 its loss (0.66) is roughly 20–40x higher than Xavier's or He's, and it needs 14 epochs to cross 90% training accuracy versus 0 (i.e., already past it after the first logged epoch) for Xavier and He. Xavier and He both converge essentially immediately for this network — consistent with theory, since both are variance-preserving initializations, and the difference between them matters most in deeper ReLU networks than the shallow 2-hidden-layer network used here. The practical takeaway holds regardless: **never use naive small-random initialization** when a principled alternative (Xavier for tanh/sigmoid, He for ReLU) is available.


## 6. Learning Rate Scheduling

A fixed learning rate forces a compromise between fast early progress and stable late-stage convergence. Learning-rate scheduling changes the step size over time — commonly starting larger and decaying it, so the optimizer can move quickly early and fine-tune later. This project implements **step decay**: `lr = initial_lr * drop^floor((1+epoch)/epochs_drop)`, equivalent to Keras' `LearningRateScheduler`. Other common schedules include exponential decay, cosine annealing, and `ReduceLROnPlateau` (decay only when validation loss stalls).

Learning-rate scheduling is evaluated jointly with early stopping and checkpointing in Section 7, since in practice these three techniques are almost always combined.


## 7. Early Stopping

Early stopping monitors validation loss during training and halts once it stops improving for a set number of epochs (the "patience"), while **model checkpointing** separately saves the weights from the best (lowest validation loss) epoch seen so far — so training can be stopped late (safely, past the point of peak performance) without losing the best model.

**Result — Plain fixed-LR training (300-epoch budget) vs. Early Stopping (patience=15) + Checkpointing + step-decay LR schedule, both from the same initialization:**

| Run | Epochs Run | Best Val Loss | Final Val Acc | Test Acc |
|---|---|---|---|---|
| Plain (fixed LR, no early stop) | 300 | 0.0441 | 0.9561 | 0.9737 |
| EarlyStop + Checkpoint + LR Schedule | 24 | 0.0441 | 0.9737 | 0.9825 |

Both runs reach the *same* best validation loss (0.0441), but the plain run reaches it at some intermediate epoch and then keeps training for 300 total epochs regardless — drifting to a worse final validation accuracy (0.9561) because it reports whatever weights exist at epoch 300, past the point of best generalization. The combined run **stops automatically at epoch 24**, using less than a tenth of the epoch budget, and — because checkpointing restores the best-epoch weights — ends with a *better* validation accuracy (0.9737) and test accuracy (0.9825) than the plain run. This is the clearest practical win demonstrated in this project: early stopping + checkpointing removes the need to guess the right fixed epoch count and protects against late-training overfitting for free.


## 8. Hyperparameter Optimization

Four one-at-a-time sweeps were run from a common baseline (`hidden=(16,8)`, He init, `lr=0.05`, `batch_size=32`, 100 epochs):

**Learning rate**

| Learning Rate | Final Val Acc | Final Val Loss | Test Acc |
|---|---|---|---|
| 0.001 | 0.9386 | 0.1281 | 0.9386 |
| 0.010 | 0.9825 | 0.0376 | 0.9649 |
| 0.050 | 0.9737 | 0.0563 | 0.9737 |
| 0.100 | 0.9649 | 0.1055 | 0.9825 |
| 0.500 | 0.6316 | 0.6582 | 0.6228 |

**Hidden-layer depth**

| Hidden Layers | # Hidden Layers | Final Val Acc | Final Val Loss | Test Acc |
|---|---|---|---|---|
| (8,) | 1 | 0.9737 | 0.0560 | 0.9825 |
| (16, 8) | 2 | 0.9737 | 0.0563 | 0.9737 |
| (32, 16, 8) | 3 | 0.9737 | 0.1445 | 0.9737 |
| (64, 32, 16, 8) | 4 | 0.9737 | 0.1370 | 0.9737 |

**Layer width** (first hidden layer size, second layer = half)

| Neurons (layer 1) | Final Val Acc | Final Val Loss | Test Acc |
|---|---|---|---|
| 4 | 0.9737 | 0.0873 | 0.9737 |
| 8 | 0.9649 | 0.0988 | 0.9737 |
| 16 | 0.9737 | 0.0563 | 0.9737 |
| 32 | 0.9649 | 0.0755 | 0.9737 |
| 64 | 0.9649 | 0.0749 | 0.9825 |

**Batch size**

| Batch Size | Final Val Acc | Final Val Loss | Test Acc |
|---|---|---|---|
| 8 | 0.9825 | 0.0794 | 0.9825 |
| 16 | 0.9737 | 0.0414 | 0.9649 |
| 32 | 0.9737 | 0.0563 | 0.9737 |
| 64 | 0.9737 | 0.0457 | 0.9649 |
| 128 | 0.9737 | 0.0390 | 0.9649 |

**Reading the sweeps:** learning rate is by far the most consequential hyperparameter here — `lr=0.5` collapses training outright (62% test accuracy, essentially random-to-poor for this task), while `lr=0.01`–`0.1` all land in a tight, healthy band (96–98%). Depth and width show a *flat* response: accuracy barely moves across 1–4 hidden layers or 4–64 neurons, though validation *loss* creeps up with extra depth (0.056 → 0.145), suggesting the deeper/wider variants are starting to overfit even though thresholded accuracy doesn't show it — validation loss is the more sensitive signal for that. Batch size has the smallest effect of the four, with all values from 8–128 landing within about 1 percentage point of each other on test accuracy.

For a dataset this size (341 training rows, 30 features), the practical conclusion is that a small network (1–2 hidden layers, 8–16 neurons) trained with a mid-range learning rate (0.01–0.05) is already close to the ceiling this data supports; additional capacity mainly adds overfitting risk rather than accuracy.


## 9. Experimental Results (Summary)

| Task | Best configuration found | Test accuracy |
|---|---|---|
| Task 1 — Baseline | `hidden=(16,8)`, He init, no dropout/BN, `lr=0.05` | 0.9825 |
| Task 2 — Dropout | No Dropout / Dropout 0.5 (tied) | 0.9825 |
| Task 3 — BatchNorm | No BatchNorm | 0.9825 |
| Task 4 — Weight init | Xavier | 0.9825 |
| Task 5 — Callbacks | EarlyStop + Checkpoint + LR schedule | 0.9825 |
| Task 6a — Learning rate | 0.1 (or 0.01, close second) | 0.9825 / 0.9649 |
| Task 6b — Depth | 1 hidden layer, `(8,)` | 0.9825 |
| Task 6c — Width | 64 neurons (marginal) | 0.9825 |
| Task 6d — Batch size | 8 | 0.9825 |

A striking pattern across nearly every sweep is that several *different, much simpler* configurations converge to the same ceiling test accuracy (0.9825, i.e. 112/114 correct on the held-out test set) as the more complex ones. This is a signature of a dataset where the class boundary is fairly linearly separable after standardization — the Breast Cancer Wisconsin features are strong, well-engineered diagnostic measurements — so most reasonably-regularized configurations converge to nearly the same solution, and hyperparameter choices mainly affect *how reliably and how fast* that solution is reached, not whether it's reached at all.


## 10. Discussion

**What mattered most.** Learning rate had the largest effect of anything tested — the only true failure case in the entire study (`lr=0.5`, 62% accuracy) came from a learning-rate choice, not from architecture, initialization, or missing regularization. Early stopping + checkpointing was the most unambiguously beneficial *technique* studied: it strictly dominated the plain fixed-epoch run (equal-or-better on every metric) while using ~8% of the epoch budget.

**What mattered less than expected.** On this dataset, dropout and batch normalization did not clearly improve validation performance, and in the BatchNorm case, arguably hurt it slightly. This is a useful, honest finding rather than a flaw in the experiment: both techniques were designed for and are best evidenced on larger, deeper networks trained on larger, higher-dimensional data (e.g., image classification). With only 341 training rows, 30 (already standardized) features, and a 2-layer network, there is limited "internal covariate shift" for BatchNorm to fix and limited co-adaptation for Dropout to break up. This is a genuinely important lesson for interns: **regularization techniques are not universally beneficial** — their value depends on dataset size, model capacity, and how much overfitting is actually present, and should be verified empirically (as done here) rather than applied by default.

**Threats to validity / limitations.**
- Single random seed per configuration in the hyperparameter sweeps (Section 8); results could shift somewhat with repeated runs and averaging, given the small validation set (114 samples ⇒ each misclassification moves accuracy by ~0.9 percentage points).
- The optimizer is plain momentum-SGD; adaptive methods (Adam, RMSProp) often change the relative sensitivity of learning rate and could shift some conclusions, particularly around how forgiving a high learning rate is.
- No L1/L2 weight decay or data augmentation was tested, both of which are standard companions to the techniques studied here.
- Results are specific to a small, mostly-linearly-separable tabular dataset; conclusions about Dropout/BatchNorm being "not very impactful" should **not** be generalized to deep CNNs on image data, where both techniques have much stronger, well-documented evidence of benefit.


## 11. Conclusion

This project trained and systematically compared ANN configurations across regularization methods (Dropout, Batch Normalization), weight initialization schemes (random, Xavier, He), training callbacks (early stopping, checkpointing, learning-rate scheduling), and a four-way hyperparameter sweep (learning rate, depth, width, batch size), on the Breast Cancer Wisconsin dataset. The baseline network overfits visibly (100% train accuracy vs. a widening validation loss after ~epoch 30), confirming the overfitting/underfitting patterns the assignment asks interns to recognize. Learning rate proved to be the single most consequential hyperparameter, with a poor choice (0.5) causing outright training failure. Early stopping combined with model checkpointing was the clearest overall win, matching or beating a much longer fixed-epoch run while using a fraction of the compute. Dropout and Batch Normalization, by contrast, showed limited benefit on this particular small tabular dataset — a reminder that regularization techniques should be chosen and validated empirically for the problem at hand, not applied automatically. He and Xavier initialization both clearly outperformed naive random initialization in convergence speed, confirming standard practice. Overall, the exercise demonstrates that disciplined experimentation — holding all but one variable fixed, tracking train *and* validation metrics, and comparing configurations on a held-out test set — is what turns "trying things" into a reliable way to build and tune a neural network.


## 12. References

1. Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. *Journal of Machine Learning Research*, 15(1), 1929–1958.
2. Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. *Proceedings of the 32nd International Conference on Machine Learning (ICML)*.
3. Glorot, X., & Bengio, Y. (2010). Understanding the Difficulty of Training Deep Feedforward Neural Networks. *Proceedings of the 13th International Conference on Artificial Intelligence and Statistics (AISTATS)*.
4. He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification. *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*.
5. Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization. *Proceedings of the 3rd International Conference on Learning Representations (ICLR)*.
6. Bergstra, J., & Bengio, Y. (2012). Random Search for Hyper-Parameter Optimization. *Journal of Machine Learning Research*, 13(1), 281–305.
7. Smith, L. N. (2017). Cyclical Learning Rates for Training Neural Networks. *IEEE Winter Conference on Applications of Computer Vision (WACV)*.
8. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
9. Prechelt, L. (1998). Early Stopping — But When? In *Neural Networks: Tricks of the Trade*, Lecture Notes in Computer Science, vol. 1524. Springer.
10. Wolberg, W. H., Street, W. N., & Mangasarian, O. L. (1995). Breast Cancer Wisconsin (Diagnostic) Data Set. UCI Machine Learning Repository. University of Wisconsin.
11. Ruder, S. (2016). An Overview of Gradient Descent Optimization Algorithms. *arXiv preprint arXiv:1609.04747*.
12. Chollet, F. (2021). *Deep Learning with Python* (2nd ed.). Manning Publications. (Reference implementation for the Keras-equivalent callback names used throughout this report: `Dropout`, `BatchNormalization`, `EarlyStopping`, `ModelCheckpoint`, `LearningRateScheduler`.)
