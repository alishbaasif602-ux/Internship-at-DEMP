# AI Intern — Week 1 Submission

This repository contains Week 1 of the AI internship: two presentations covering Machine
Learning fundamentals, along with the code that implements and tests them.


## 📁 What's Included

| File | Description |
|---|---|
| `Supervised_Learning_Models_Simplified.pptx` | Presentation on 4 core supervised learning models |
| `Optimization_Techniques_Simplified.pptx` | Presentation on 6 optimization algorithms |
| `Optimization_Techniques_Code.ipynb` | Code implementing the optimizers from scratch |
| `Financial_ML_Models_Assignment.ipynb` | Code applying SL models to financial use cases (Set 1) |
| `Four_Models_Financial_Use_Cases.ipynb` | Code applying SL models to financial use cases (Set 2) |

Both presentations were built in **PowerPoint** (slides), with equations designed for
clarity and diagrams included alongside each concept.



## 1️⃣ Supervised Learning Models

### Linear Regression
**Definition:** An algorithm that predicts a continuous number by fitting the straight line
that best matches the data.
**Example:** Predicting a house's price based on its size in square feet — as size increases,
price increases roughly in a straight-line pattern.

### Logistic Regression
**Definition:** An algorithm that predicts a binary outcome (yes/no) by converting a score
into a probability using the sigmoid function.
**Example:** Predicting whether a bank loan applicant will default (Yes/No) based on their
income and credit score.

### Decision Trees
**Definition:** A model that makes predictions by asking a series of yes/no questions about
the data, splitting it step by step until it reaches a decision.
**Example:** Deciding whether to approve a credit card — "Is income > 50k? → Is credit score
> 700? → Approve/Reject."

### Support Vector Machines (SVM)
**Definition:** A classification algorithm that finds the best boundary (hyperplane) to
separate two classes with the widest possible gap between them.
**Example:** Separating emails into "Spam" and "Not Spam" based on word frequency features.


## 2️⃣ Optimization Techniques

### Batch Gradient Descent
**Definition:** Updates the model's weights using the entire dataset at once, one update per
full pass through the data.
**Example:** Like reviewing every single customer review before deciding how to adjust a
product — thorough, but slow.

### Stochastic Gradient Descent (SGD)
**Definition:** Updates the weights using just one random data point at a time.
**Example:** Adjusting a product based on just the very last customer review you read —
fast, but reactive and inconsistent.

### Mini-Batch Gradient Descent
**Definition:** Updates the weights using a small group (batch) of data points at a time —
a balance between Batch GD and SGD.
**Example:** Adjusting a product after reading a batch of 32 reviews at once — quick and
reasonably reliable.

### Momentum
**Definition:** Speeds up gradient descent by remembering the direction of previous updates,
so it doesn't get sidetracked by small fluctuations.
**Example:** Like a ball rolling downhill — it keeps rolling in the same general direction
even if the ground gets slightly bumpy.

### RMSProp
**Definition:** Gives each parameter its own adjustable step size, based on how much that
parameter's gradient has changed recently.
**Example:** Like slowing down when driving over rough terrain but speeding up on a smooth
road — the "step size" adapts to conditions.

### Adam
**Definition:** Combines Momentum and RMSProp together — the most commonly used optimizer in
deep learning today.
**Example:** Like a smart cruise control that adjusts both direction and speed automatically
based on the road ahead — fast and stable.



## 💻 Coding Part (Brief)

- **`Optimization_Techniques_Code.ipynb`** — Implements all four optimizers (Batch GD, SGD,
  Mini-Batch GD, Adam) from scratch in Python/NumPy on a simple dummy dataset, then compares
  their final accuracy (loss). Result: Adam performed best (lowest loss), followed closely by
  Mini-Batch GD.

- **`Financial_ML_Models_Assignment.ipynb`** and **`Four_Models_Financial_Use_Cases.ipynb`**
  — Apply the supervised learning models (Linear/Logistic Regression, Random Forest, Naive
  Bayes, KNN) to realistic financial scenarios (stock price prediction, loan approval, fraud
  detection, churn prediction, etc.) using synthetic datasets, and evaluate each model's
  accuracy using standard metrics (R², Accuracy, ROC-AUC).

Both notebooks are ready to run in Google Colab — just upload and run all cells.



## 🛠 Tools Used
- **PowerPoint** — for building both presentations
- **Python** (NumPy, Pandas, scikit-learn, Matplotlib) — for all code implementation
- **Google Colab / Jupyter** — for running and testing the notebooks

## ✅ Summary

This week covered both the **theory** and the **practical implementation** of core Machine
Learning concepts:

- Studied and presented **4 Supervised Learning models** (Linear Regression, Logistic
  Regression, Decision Trees, SVM) — what each one does, with simple real-world examples.
- Studied and presented **6 Optimization Techniques** (Batch GD, SGD, Mini-Batch GD, Momentum,
  RMSProp, Adam) — how each one updates a model's weights during training.
- **Implemented all 4 optimizers from scratch** in Python and compared their accuracy — Adam
  and Mini-Batch GD performed best.
- **Applied the supervised learning models to 8 financial use cases** (stock price prediction,
  loan approval/default, fraud detection, churn prediction, insurance premium, and property
  valuation), evaluating each with proper accuracy metrics (R², Accuracy, ROC-AUC).

Overall, this week's work connects the mathematical theory (presentations) directly to working,
tested code (notebooks) across both classification and regression problems.
