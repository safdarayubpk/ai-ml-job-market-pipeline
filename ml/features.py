from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import spmatrix
from scraper.parser import SKILL_KEYWORDS


def build_tfidf_matrix(descriptions: list[str]) -> tuple[spmatrix, TfidfVectorizer]:
    """
    Fit TF-IDF on job descriptions.
    Returns (sparse matrix, fitted vectorizer).
    min_df=1 used for small datasets; increase to 2 once you have 50+ jobs.
    """
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        min_df=1,
        ngram_range=(1, 2),
    )
    matrix = vectorizer.fit_transform(descriptions)
    return matrix, vectorizer


def compute_skill_frequencies(descriptions: list[str]) -> dict[str, float]:
    """
    Return skill keyword frequencies as percentage of jobs mentioning each skill.
    Skills with 0 mentions are excluded. Result is sorted descending by frequency.
    """
    total = len(descriptions)
    if total == 0:
        return {}
    counts: dict[str, int] = {}
    for desc in descriptions:
        desc_lower = desc.lower()
        for skill in SKILL_KEYWORDS:
            if skill in desc_lower:
                counts[skill] = counts.get(skill, 0) + 1
    return {
        skill: round(count / total * 100, 1)
        for skill, count in sorted(counts.items(), key=lambda x: -x[1])
        if count > 0
    }
