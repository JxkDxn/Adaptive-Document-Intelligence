"""Lightweight TF-IDF + LinearSVC document classifier.

Features are derived only from document text. Filename and other metadata
are never used as model inputs.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

SEED = 42
C_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)


class LocalDocumentClassifier:
    """Train and predict document class labels from text content only."""

    def __init__(self, random_state: int = SEED) -> None:
        self.random_state = random_state
        self.pipeline_: Pipeline | None = None
        self.classes_ = None
        self.C_: float | None = None

    def _make_pipeline(self, C: float) -> Pipeline:
        features = FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        analyzer="word",
                        ngram_range=(1, 2),
                        min_df=2,
                        max_df=0.95,
                        sublinear_tf=True,
                        lowercase=True,
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=2,
                        max_df=0.95,
                        sublinear_tf=True,
                        lowercase=True,
                    ),
                ),
            ]
        )
        clf = LinearSVC(
            C=C,
            random_state=self.random_state,
            max_iter=8000,
            dual="auto",
        )
        return Pipeline([("features", features), ("clf", clf)])

    def fit(
        self,
        texts: list[str],
        labels: list[str],
        validation_texts: list[str] | None = None,
        validation_labels: list[str] | None = None,
    ) -> LocalDocumentClassifier:
        """Fit on training text only. Validation is used only to choose C."""
        if validation_texts is not None and validation_labels is not None and len(validation_texts) > 0:
            best_c = C_GRID[0]
            best_score = -1.0
            for C in C_GRID:
                pipeline = self._make_pipeline(C)
                pipeline.fit(texts, labels)
                pred = pipeline.predict(validation_texts)
                score = float(f1_score(validation_labels, pred, average="macro", zero_division=0))
                if score > best_score:
                    best_score = score
                    best_c = C
            self.C_ = best_c
        else:
            self.C_ = 1.0

        self.pipeline_ = self._make_pipeline(self.C_)
        self.pipeline_.fit(texts, labels)
        self.classes_ = list(self.pipeline_.named_steps["clf"].classes_)
        return self

    def predict(self, texts: list[str]) -> list[str]:
        if self.pipeline_ is None:
            raise RuntimeError("LocalDocumentClassifier must be fit before predict.")
        return list(self.pipeline_.predict(texts))
