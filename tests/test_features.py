from ml.features import build_tfidf_matrix, compute_skill_frequencies

SAMPLE_DESCRIPTIONS = [
    "We need a Python developer with strong PyTorch skills for LLM work.",
    "Looking for a machine learning engineer with scikit-learn and TensorFlow experience.",
    "Seeking a data scientist who knows Python and SQL for our analytics team.",
    "ML Engineer with LangChain, OpenAI, and RAG pipeline experience required.",
    "Backend developer with Python and PostgreSQL, some ML exposure preferred.",
]


def test_build_tfidf_matrix_row_count():
    matrix, _ = build_tfidf_matrix(SAMPLE_DESCRIPTIONS)
    assert matrix.shape[0] == len(SAMPLE_DESCRIPTIONS)


def test_build_tfidf_matrix_max_features():
    matrix, _ = build_tfidf_matrix(SAMPLE_DESCRIPTIONS)
    assert matrix.shape[1] <= 5000


def test_build_tfidf_matrix_is_sparse():
    from scipy.sparse import issparse
    matrix, _ = build_tfidf_matrix(SAMPLE_DESCRIPTIONS)
    assert issparse(matrix)


def test_compute_skill_frequencies_python_present():
    freqs = compute_skill_frequencies(SAMPLE_DESCRIPTIONS)
    assert "python" in freqs
    assert freqs["python"] == 60.0  # 3 of 5 descriptions mention python


def test_compute_skill_frequencies_sorted_descending():
    freqs = compute_skill_frequencies(SAMPLE_DESCRIPTIONS)
    values = list(freqs.values())
    assert values == sorted(values, reverse=True)


def test_compute_skill_frequencies_empty_list():
    assert compute_skill_frequencies([]) == {}


def test_compute_skill_frequencies_excludes_zero_count_skills():
    freqs = compute_skill_frequencies(["This description mentions nothing relevant."])
    assert all(v > 0 for v in freqs.values())
