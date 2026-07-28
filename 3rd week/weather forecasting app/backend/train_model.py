import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv("../dataset/weather dataset.csv")

# Features and Target
X = df.drop("Weather Type", axis=1)
y = df["Weather Type"]

# Categorical Columns
categorical_cols = ["Cloud Cover", "Season", "Location"]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ],
    remainder="passthrough"
)

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ===========================
# Logistic Regression Model
# ===========================
logistic_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

logistic_model.fit(X_train, y_train)

log_pred = logistic_model.predict(X_test)

log_acc = accuracy_score(y_test, log_pred)

print(f"Logistic Regression Accuracy: {log_acc:.2%}")

# ===========================
# SVM Model
# ===========================
svm_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", SVC())
])

svm_model.fit(X_train, y_train)

svm_pred = svm_model.predict(X_test)

svm_acc = accuracy_score(y_test, svm_pred)

print(f"SVM Accuracy: {svm_acc:.2%}")

# ===========================
# KNN Model
# ===========================
knn_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", KNeighborsClassifier(n_neighbors=5))
])

knn_model.fit(X_train, y_train)

knn_pred = knn_model.predict(X_test)

knn_acc = accuracy_score(y_test, knn_pred)

print(f"KNN Accuracy: {knn_acc:.2%}")

# ===========================
# Save Models
# ===========================
joblib.dump(logistic_model, "models/logistic_model.pkl")
joblib.dump(svm_model, "models/svm_model.pkl")
joblib.dump(knn_model, "models/knn_model.pkl")

print("\nAll models saved successfully!")