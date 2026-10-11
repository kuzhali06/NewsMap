"""
TF-IDF Feature Extraction Module for NewsMap (Phase 5, Module 2).

Transforms preprocessed, validated news article text into a sparse TF-IDF numerical matrix
for unsupervised topic discovery and clustering.

Key Architectural Guarantees:
1. Reuses Module 1 input validation to guarantee data integrity and strict label decoupling.
2. Preserves sparse matrix format (scipy.sparse.csr_matrix) throughout to conserve memory.
3. Maps every matrix row directly back to its unique article_id.
4. Uses deterministic, configurable vectorizer settings.
5. Saves fitted vectorizer using joblib for downstream inference and API services.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, isspmatrix_csr
from sklearn.feature_extraction.text import TfidfVectorizer

from backend.app.ml.input_validation import validate_corpus_for_tfidf
from backend.app.services.data_loader import get_project_root, load_processed_articles

# Default baseline configuration tailored for the BBC News corpus
DEFAULT_MIN_DF = 3
DEFAULT_MAX_DF = 0.7
DEFAULT_MAX_FEATURES = 5000
DEFAULT_NGRAM_RANGE = (1, 2)
DEFAULT_SUBLINEAR_TF = True
DEFAULT_STOP_WORDS = "english"
DEFAULT_TOKEN_PATTERN = r"(?u)\b[a-zA-Z0-9]+(?:[-'][a-zA-Z0-9]+)*\b"
DEFAULT_NORM = "l2"


def extract_tfidf_features(
    corpus: Union[pd.DataFrame, List[str]],
    article_ids: Optional[List[int]] = None,
    text_column: str = "cleaned_text",
    id_column: str = "article_id",
    max_features: Optional[int] = DEFAULT_MAX_FEATURES,
    min_df: Union[int, float] = DEFAULT_MIN_DF,
    max_df: Union[int, float] = DEFAULT_MAX_DF,
    ngram_range: Tuple[int, int] = DEFAULT_NGRAM_RANGE,
    sublinear_tf: bool = DEFAULT_SUBLINEAR_TF,
    stop_words: Optional[Union[str, List[str]]] = DEFAULT_STOP_WORDS,
    token_pattern: str = DEFAULT_TOKEN_PATTERN,
    norm: str = DEFAULT_NORM,
) -> Tuple[TfidfVectorizer, csr_matrix, Dict[str, Any]]:
    """
    Fit and transform cleaned article texts into a sparse TF-IDF matrix.

    Args:
        corpus: Either a pandas DataFrame containing cleaned articles (validated),
                or a list of preprocessed text strings.
        article_ids: Optional list of integer IDs if corpus is a list of strings.
        text_column: Column name containing text if corpus is a DataFrame.
        id_column: Column name containing IDs if corpus is a DataFrame.
        max_features: Maximum vocabulary size ordered by term frequency. Defaults to 5,000.
        min_df: Minimum document frequency (integer count or float fraction). Defaults to 3.
        max_df: Maximum document frequency (float fraction or integer count). Defaults to 0.7.
        ngram_range: Lower and upper boundary of n-grams. Defaults to (1, 2) for unigrams+bigrams.
        sublinear_tf: Apply sublinear scaling 1 + log(tf) to damp long-document bias. Defaults to True.
        stop_words: Stop words filter ('english' or list). Defaults to 'english'.
        token_pattern: Regular expression defining token boundaries. Preserves compound hyphens.
        norm: Normalization norm ('l1', 'l2', or None). Defaults to 'l2'.

    Returns:
        Tuple of:
        - Fitted TfidfVectorizer instance.
        - Sparse TF-IDF feature matrix (scipy.sparse.csr_matrix).
        - Metadata dictionary detailing matrix shape, vocabulary, row-to-ID mappings, and sparsity.

    Raises:
        TypeError: If input corpus format is unsupported.
        ValueError: If input is empty, malformed, leaks category labels, or yields an empty vocabulary.
    """
    # 1. Input parsing and validation
    if isinstance(corpus, pd.DataFrame):
        # Reuse Module 1 validation: enforces schema, no nulls, unique IDs, NO category leakage
        validated_df = validate_corpus_for_tfidf(corpus, text_col=text_column, id_col=id_column)
        texts = validated_df[text_column].tolist()
        resolved_ids = validated_df[id_column].astype(int).tolist()
    elif isinstance(corpus, (list, tuple)):
        if not corpus:
            raise ValueError("Corpus list is empty. At least one text record is required.")
        for idx, item in enumerate(corpus):
            if item is None or not isinstance(item, str) or not item.strip():
                raise ValueError(
                    f"Corpus record at index {idx} is invalid, null, or empty string."
                )
        texts = list(corpus)
        if article_ids is None:
            resolved_ids = list(range(len(texts)))
        else:
            if len(article_ids) != len(texts):
                raise ValueError(
                    f"Mismatch between number of article_ids ({len(article_ids)}) "
                    f"and corpus texts ({len(texts)})."
                )
            if len(set(article_ids)) != len(article_ids):
                raise ValueError("Provided article_ids list contains duplicate IDs.")
            resolved_ids = [int(aid) for aid in article_ids]
    else:
        raise TypeError(
            f"Unsupported corpus input type '{type(corpus).__name__}'. "
            f"Expected pandas DataFrame or list of strings."
        )

    # 2. Configure TfidfVectorizer
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
        stop_words=stop_words,
        token_pattern=token_pattern,
        norm=norm,
    )

    # 3. Fit and transform corpus
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
    except ValueError as exc:
        raise ValueError(
            f"TF-IDF vectorization failed: {exc}. Verify that the corpus is not empty "
            f"and that min_df/stop_words settings do not eliminate the entire vocabulary."
        ) from exc

    # Ensure sparse CSR matrix format is preserved
    if not isspmatrix_csr(tfidf_matrix):
        tfidf_matrix = tfidf_matrix.tocsr()

    # 4. Extract matrix metrics & verify row integrity
    n_docs, n_features = tfidf_matrix.shape
    if n_features == 0:
        raise ValueError("TF-IDF extraction resulted in an empty vocabulary (0 features).")

    nnz = int(tfidf_matrix.nnz)
    total_elements = n_docs * n_features
    sparsity = float(1.0 - (nnz / total_elements)) if total_elements > 0 else 0.0

    # Check for empty feature vectors (documents with all-zero weights)
    row_sums = np.array(tfidf_matrix.sum(axis=1)).ravel()
    empty_doc_indices = [int(i) for i in np.where(row_sums == 0)[0]]
    empty_doc_article_ids = [resolved_ids[i] for i in empty_doc_indices]

    feature_names = vectorizer.get_feature_names_out().tolist()

    # 5. Compile metadata dictionary
    metadata: Dict[str, Any] = {
        "n_documents": n_docs,
        "n_features": n_features,
        "matrix_shape": (n_docs, n_features),
        "matrix_format": type(tfidf_matrix).__name__,
        "nnz": nnz,
        "sparsity": sparsity,
        "density": float(1.0 - sparsity),
        "article_ids": resolved_ids,
        "feature_names": feature_names,
        "empty_vector_count": len(empty_doc_indices),
        "empty_vector_indices": empty_doc_indices,
        "empty_vector_article_ids": empty_doc_article_ids,
        "config": {
            "max_features": max_features,
            "min_df": min_df,
            "max_df": max_df,
            "ngram_range": ngram_range,
            "sublinear_tf": sublinear_tf,
            "stop_words": stop_words,
            "token_pattern": token_pattern,
            "norm": norm,
        },
    }

    return vectorizer, tfidf_matrix, metadata


def get_top_terms_for_document(
    tfidf_matrix: csr_matrix,
    feature_names: List[str],
    doc_index: int,
    top_n: int = 5,
) -> List[Tuple[str, float]]:
    """
    Extract the highest-weighted TF-IDF terms for a single document.

    Operates directly on the sparse row without converting the full matrix to a dense array.

    Args:
        tfidf_matrix: Sparse CSR matrix of TF-IDF weights.
        feature_names: Ordered list of feature terms.
        doc_index: Row index of the document to inspect.
        top_n: Number of top terms to return. Defaults to 5.

    Returns:
        List of tuples: (term, weight) sorted by weight in descending order.
    """
    if doc_index < 0 or doc_index >= tfidf_matrix.shape[0]:
        raise IndexError(
            f"doc_index {doc_index} is out of bounds for matrix with {tfidf_matrix.shape[0]} rows."
        )

    # Extract non-zero column indices and values directly from sparse CSR row
    row = tfidf_matrix.getrow(doc_index)
    col_indices = row.indices
    data_values = row.data

    if len(data_values) == 0:
        return []

    # Sort top indices by weight descending
    top_sort = np.argsort(data_values)[::-1][:top_n]
    return [
        (feature_names[col_indices[i]], round(float(data_values[i]), 4))
        for i in top_sort
    ]


def save_tfidf_vectorizer(
    vectorizer: TfidfVectorizer,
    output_path: Optional[Union[str, Path]] = None,
    project_root: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Save the fitted TfidfVectorizer to disk using joblib.

    Security note: Persisted Python objects (joblib/pickle) should only be loaded
    from trusted project artifacts to prevent arbitrary code execution.

    Args:
        vectorizer: Fitted TfidfVectorizer instance.
        output_path: Optional explicit file path. Defaults to backend/models/tfidf_vectorizer.joblib.
        project_root: Optional project root path.

    Returns:
        Path to the saved model file.
    """
    if output_path is None:
        root = get_project_root(project_root)
        target_file = root / "backend" / "models" / "tfidf_vectorizer.joblib"
    else:
        target_file = Path(output_path).resolve()

    target_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, target_file)
    return target_file


def load_tfidf_vectorizer(
    model_path: Optional[Union[str, Path]] = None,
    project_root: Optional[Union[str, Path]] = None,
) -> TfidfVectorizer:
    """
    Load a fitted TfidfVectorizer from disk using joblib.

    Security note: Persisted Python objects should only be loaded from trusted project artifacts.

    Args:
        model_path: Optional explicit path. Defaults to backend/models/tfidf_vectorizer.joblib.
        project_root: Optional project root path.

    Returns:
        Loaded TfidfVectorizer instance.
    """
    if model_path is None:
        root = get_project_root(project_root)
        source_file = root / "backend" / "models" / "tfidf_vectorizer.joblib"
    else:
        source_file = Path(model_path).resolve()

    if not source_file.exists():
        raise FileNotFoundError(
            f"TF-IDF vectorizer artifact not found at: '{source_file}'. "
            f"Please run feature extraction to generate the vectorizer first."
        )

    return joblib.load(source_file)


def main() -> None:
    """
    CLI / module entry point to execute Phase 5 Module 2 Feature Extraction.
    """
    print("=" * 70)
    print("NewsMap — Phase 5, Module 2: TF-IDF Feature Extraction")
    print("=" * 70)

    root = get_project_root()
    print(f"Project root: {root}")

    # 1. Load preprocessed articles
    print("\n[1/4] Loading and validating cleaned article corpus...")
    articles_df = load_processed_articles(root)
    print(f"  -> Cleaned articles loaded: {len(articles_df):,} records")
    print(f"  -> Schema: {list(articles_df.columns)}")

    # 2. Extract TF-IDF features
    print("\n[2/4] Fitting TF-IDF Vectorizer (Baseline Config)...")
    print(f"  -> Parameters: min_df={DEFAULT_MIN_DF}, max_df={DEFAULT_MAX_DF}, "
          f"max_features={DEFAULT_MAX_FEATURES}, ngram_range={DEFAULT_NGRAM_RANGE}, sublinear_tf=True")

    vectorizer, tfidf_matrix, meta = extract_tfidf_features(articles_df)

    # 3. Report detailed extraction metrics
    print("\n[3/4] Extraction Metrics & Matrix Characteristics:")
    print(f"  -> Documents processed:         {meta['n_documents']:,}")
    print(f"  -> Vocabulary size (features):  {meta['n_features']:,}")
    print(f"  -> Matrix dimensions:           {meta['matrix_shape']}")
    print(f"  -> Sparse matrix format:        {meta['matrix_format']} ({tfidf_matrix.format})")
    print(f"  -> Non-zero entries (NNZ):      {meta['nnz']:,}")
    print(f"  -> Matrix sparsity:             {meta['sparsity'] * 100:.2f}% (density: {meta['density'] * 100:.2f}%)")
    print(f"  -> Empty feature vectors:       {meta['empty_vector_count']}")

    # Sample top terms for 5 representative articles
    feature_names = meta["feature_names"]
    print("\nTop-weighted terms for sample articles:")
    sample_indices = [0, 5, 20, 50, 100]
    for idx in sample_indices:
        aid = meta["article_ids"][idx]
        top_terms = get_top_terms_for_document(tfidf_matrix, feature_names, idx, top_n=5)
        text_snippet = articles_df.iloc[idx]["cleaned_text"][:85]
        print(f"  - Article ID {aid} (Row {idx}):")
        print(f"    Snippet: \"{text_snippet}...\"")
        print(f"    Top Terms: {top_terms}")

    # 4. Save fitted vectorizer
    print("\n[4/4] Persisting vectorizer artifact...")
    saved_path = save_tfidf_vectorizer(vectorizer, project_root=root)
    print(f"  -> Saved fitted vectorizer to: {saved_path}")
    print("  -> Note: Persisted Python objects should only be loaded from trusted project artifacts.")

    print("\n" + "=" * 70)
    print("Phase 5 Module 2 Feature Extraction completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
