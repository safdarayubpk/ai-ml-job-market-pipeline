import numpy as np
from scipy.sparse import csr_matrix
from ml.classifier import label_seniority, compare_classifiers, train_best_classifier, predict_seniority


def test_label_seniority_senior_title():
    assert label_seniority("Senior ML Engineer", "") == "senior"


def test_label_seniority_sr_abbreviation():
    assert label_seniority("Sr. AI Engineer", "") == "senior"


def test_label_seniority_lead():
    assert label_seniority("Lead ML Engineer", "") == "senior"


def test_label_seniority_junior_title():
    assert label_seniority("Junior Data Scientist", "") == "junior"


def test_label_seniority_jr_abbreviation():
    assert label_seniority("Jr. NLP Engineer", "") == "junior"


def test_label_seniority_entry_level_in_description():
    assert label_seniority("Data Scientist", "This is an entry level position.") == "junior"


def test_label_seniority_mid_is_default():
    assert label_seniority("ML Engineer", "Python and ML experience required.") == "mid"


def test_compare_classifiers_returns_three_models():
    rng = np.random.RandomState(42)
    X = csr_matrix(rng.rand(60, 30))
    y = ["junior"] * 20 + ["mid"] * 20 + ["senior"] * 20
    scores = compare_classifiers(X, y)
    assert set(scores.keys()) == {"LogisticRegression", "LinearSVC", "RandomForest"}


def test_compare_classifiers_scores_between_0_and_1():
    rng = np.random.RandomState(42)
    X = csr_matrix(rng.rand(60, 30))
    y = ["junior"] * 20 + ["mid"] * 20 + ["senior"] * 20
    scores = compare_classifiers(X, y)
    for name, score in scores.items():
        assert 0.0 <= score <= 1.0, f"{name} score {score} out of range"


def test_predict_seniority_returns_labels():
    rng = np.random.RandomState(42)
    X = csr_matrix(rng.rand(60, 30))
    y = ["junior"] * 20 + ["mid"] * 20 + ["senior"] * 20
    model, le = train_best_classifier(X, y)
    predictions = predict_seniority(model, le, X)
    assert len(predictions) == 60
    assert all(p in {"junior", "mid", "senior"} for p in predictions)
