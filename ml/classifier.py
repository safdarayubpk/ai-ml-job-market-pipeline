import re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import spmatrix

SENIOR_RE = re.compile(
    r"\b(senior|sr\.?|lead|principal|staff|head of|vp|director|architect)\b", re.I
)
JUNIOR_RE = re.compile(
    r"\b(junior|jr\.?|entry[\s\-]?level|graduate|intern|associate|0[\s\-]?1 years?)\b", re.I
)


def label_seniority(title: str, description: str) -> str:
    """
    Rule-based seniority classifier using title + description patterns.
    Title takes precedence; description is checked only if title is neutral.
    Returns 'junior', 'mid', or 'senior'.
    """
    if SENIOR_RE.search(title):
        return "senior"
    if JUNIOR_RE.search(title):
        return "junior"
    if SENIOR_RE.search(description):
        return "senior"
    if JUNIOR_RE.search(description):
        return "junior"
    return "mid"


def compare_classifiers(X: spmatrix, y: list[str]) -> dict[str, float]:
    """
    Evaluate LR, LinearSVC, and RandomForest on (X, y) using 5-fold CV.
    Returns {model_name: mean_f1_macro_score}.
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "LinearSVC": LinearSVC(max_iter=2000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    }
    results: dict[str, float] = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y_enc, cv=5, scoring="f1_macro")
        results[name] = round(float(scores.mean()), 3)
    return results


def train_best_classifier(
    X: spmatrix, y: list[str]
) -> tuple[LogisticRegression, LabelEncoder]:
    """Train LogisticRegression (best for text tasks) and return (model, encoder)."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y_enc)
    return model, le


def predict_seniority(
    model: LogisticRegression, le: LabelEncoder, X: spmatrix
) -> list[str]:
    """Return predicted seniority label strings for each row in X."""
    y_pred = model.predict(X)
    return list(le.inverse_transform(y_pred))
