import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


def train_models(X, y):

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000))
        ]),

        "Decision Tree": DecisionTreeClassifier(random_state=42),

        "k-NN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5))
        ]),

        "Naive Bayes": GaussianNB(),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ),

        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(probability=True, random_state=42))
        ]),

        "Neural Net (MLP)": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=1000,
                random_state=42
            ))
        ]),
    }

    multi_class = len(np.unique(y)) > 2

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    MODELS = {}
    MODEL_METRICS = {}

    best_acc = -1
    best_name = None
    best_confusion = None

    for name, model in models.items():
        print(f"🔹 Training {name}")

        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Probabilities (if available)
        y_prob = None
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)

        # ========== METRICS ==========
        acc = accuracy_score(y_test, y_pred)

        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # SAFE ROC AUC FOR BOTH BINARY + MULTICLASS
        roc = None
        try:
            if y_prob is not None:
                if multi_class:
                    roc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
                else:
                    roc = roc_auc_score(y_test, y_prob[:, 1])
        except:
            roc = None

        # Confusion Matrix (for ALL classes)
        cm = confusion_matrix(y_test, y_pred)

        MODEL_METRICS[name] = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "roc_auc": float(roc) if roc is not None else None,
            "confusion_matrix": {
                "matrix": cm.tolist(),
                "classes": len(cm)
            },
        }

        MODELS[name] = model

        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_confusion = cm.tolist()

    return MODELS, MODEL_METRICS, best_name, best_confusion
