# NewsMap — Unsupervised News Topic Discovery & Related-Story Detection

An end-to-end Machine Learning and Natural Language Processing application for unsupervised news topic clustering and related-story discovery using the BBC News dataset.

---

## Project Structure

```
D:\NewsMap\
|-- backend\
|   |-- app\
|   |   |-- __init__.py
|   |   |-- core\
|   |   |   `-- config.py
|   |   |-- ml\
|   |   |   |-- __init__.py
|   |   |   |-- preprocessing.py          # Phase 4: Text cleaning and tokenization
|   |   |   |-- input_validation.py       # Phase 5, Mod 1: Corpus & label validation
|   |   |   |-- feature_extraction.py     # Phase 5, Mod 2: TF-IDF feature extraction
|   |   |   `-- clustering.py             # Phase 5, Mod 3: K-Means topic clustering
|   |   `-- services\
|   |       |-- __init__.py
|   |       |-- data_loader.py            # Dataset loading and export utilities
|   |       `-- related_stories.py        # Phase 5, Mod 4: Cosine similarity retrieval
|   |-- data\
|   |   |-- raw\                          # Raw BBC News CSVs
|   |   |   |-- BBC News Train.csv
|   |   |   |-- BBC News Test.csv
|   |   |   `-- BBC News Sample Solution.csv
|   |   `-- processed\                    # Processed datasets
|   |       |-- articles_clean.csv        # Deduplicated unlabeled corpus (2,123 articles)
|   |       |-- evaluation_labels.csv     # Ground-truth evaluation labels (2,225 records)
|   |       |-- cluster_assignments.csv   # Cluster ID per article_id (2,123 rows)
|   |       `-- cluster_summaries.json    # Interpretable topic summaries & representative stories
|   |-- models\                           # Fitted model artifacts
|   |   |-- tfidf_vectorizer.joblib       # Fitted TfidfVectorizer (5,000 features)
|   |   `-- kmeans_model.joblib           # Fitted KMeans model (k=5)
|   |-- tests\                            # Unit and integration test suites
|   |   |-- __init__.py
|   |   |-- test_preprocessing.py         # Phase 4 tests (18 tests)
|   |   |-- test_input_validation.py      # Phase 5, Mod 1 tests (14 tests)
|   |   |-- test_feature_extraction.py    # Phase 5, Mod 2 tests (11 tests)
|   |   |-- test_clustering.py            # Phase 5, Mod 3 tests (12 tests)
|   |   `-- test_related_stories.py       # Phase 5, Mod 4 tests (13 tests)
|   `-- run_phase4.py                     # Phase 4 execution runner script
|-- docs/                                 # Project documentation
|-- frontend/                             # React frontend (later phase)
|-- inspect_dataset.py                    # Initial raw inspection script
`-- README.md                             # Project documentation
```

---

## Phase 4: Data Cleaning and Text Preprocessing

Phase 4 implements data ingestion, quality inspection, text cleaning, and dataset assembly.

### Files Implemented in Phase 4

1. `backend/app/ml/preprocessing.py`:
   - `clean_text(text, preserve_case, preserve_numbers)`: Reusable, deterministic text-cleaning function.
     - Preserves intra-word hyphens (`peer-to-peer`, `cost-effective`, `t-mobile`).
     - Preserves intra-word contractions and negations (`didn't`, `won't`, `not`, `never`).
     - Preserves decimal numbers (`3.8`, `95.5`), currency symbols (`£`, `$`), and percentages (`%`).
     - Removes noisy structural punctuation (`()`, `[]`, `{}`, `;`, `:`, `!`, `?`, `*`, `#`, `@`, `&`).
     - Avoids indiscriminate stop-word deletion or aggressive stemming.
   - `clean_series(...)`: Batch cleaning for pandas Series.
   - `extract_tokens(...)` & `get_text_statistics(...)`: Utility helpers for word and character counts.

2. `backend/app/services/data_loader.py`:
   - `get_project_root(...)`: Automatically locates project root dynamically without hardcoded absolute paths.
   - `load_training_data(...)`, `load_test_data(...)`, `load_sample_solution(...)`: Robust CSV loaders validating mandatory schemas (`ArticleId`, `Text`, `Category`).
   - `inspect_dataset_quality(...)`: Audits missing values, empty texts, repeated IDs, and duplicate texts.
   - `prepare_article_collection(...)`: Assembles the combined unlabeled corpus with explicit deduplication (`keep_first`).
   - `save_processed_articles(...)` & `save_evaluation_labels(...)`: Exports processed data.

3. `backend/tests/test_preprocessing.py`:
   - 18 unit tests covering preprocessing edge cases, whitespace, punctuation, duplicate detection, and schema validation.

---

## Phase 5 — Module 1: Environment and Input Validation

Module 1 verifies environment compatibility, audits processed artifacts, and establishes a validated input interface for unsupervised modeling.

### Features Implemented in Phase 5, Module 1

1. `backend/app/ml/input_validation.py`:
   - `validate_corpus_for_tfidf(df)`: Validates that an article corpus is structurally sound before passing to TF-IDF.
     - Confirms required columns (`article_id`, `cleaned_text`).
     - Rejects null values, empty strings, and duplicate article IDs.
     - **Strict Label Decoupling:** Rejects any DataFrame containing category/target labels (`category`, `label`, `target`) to guarantee unsupervised isolation.
   - `validate_evaluation_labels(labels_df, articles_df)`: Audits ground-truth category labels.
     - Confirms schema (`article_id`, `category`).
     - Assesses overlap and coverage against articles without assuming every article must be labeled.
     - Tracks extra labels (e.g. from deduplicated articles).
   - `prepare_validated_tfidf_inputs(...)`: High-level loader and validation pipeline providing verified inputs for Module 2.

2. `backend/app/services/data_loader.py` (Additions):
   - `load_processed_articles(...)`: Loads and validates `backend/data/processed/articles_clean.csv`.
   - `load_processed_evaluation_labels(...)`: Loads and validates `backend/data/processed/evaluation_labels.csv`.

3. `backend/tests/test_input_validation.py`:
   - 14 focused unit tests covering corpus validation, type errors, null/empty text rejection, duplicate ID rejection, category leakage prevention, partial label coverage, extra labels tracking, and integration with real processed disk files.

---

## Phase 5 — Module 2: TF-IDF Feature Extraction

Module 2 transforms the validated, cleaned text corpus into a sparse TF-IDF matrix for unsupervised topic clustering.

### Key Capabilities

1. `backend/app/ml/feature_extraction.py`:
   - `extract_tfidf_features(corpus, ...)`: Fits and transforms the corpus into a sparse `scipy.sparse.csr_matrix`.
     - Integrates seamlessly with Module 1's validation interface.
     - Never converts sparse matrices to dense arrays, conserving system memory.
     - Returns complete metadata mapping row index `i` directly to its unique `article_id`.
     - Evaluates matrix sparsity and checks for any empty feature vectors.
   - `get_top_terms_for_document(tfidf_matrix, feature_names, doc_index, top_n)`: Inspects highest-weighted terms per document directly from sparse rows.
   - `save_tfidf_vectorizer(vectorizer, ...)` & `load_tfidf_vectorizer(...)`: Persists the fitted vectorizer to `backend/models/tfidf_vectorizer.joblib`.
     - *Security notice: Persisted Python objects (joblib/pickle) should only be loaded from trusted project artifacts.*

2. `backend/tests/test_feature_extraction.py`:
   - 11 unit and integration tests covering matrix creation, row-to-ID alignment, deterministic reproducibility, invalid input rejection, configurable settings, CSR format preservation, empty vocabulary handling, category label isolation, and joblib serialization.

### Baseline Vectorizer Configuration & Trade-offs

| Parameter | Selected Baseline | Rationale & Trade-offs |
| :--- | :--- | :--- |
| `ngram_range` | `(1, 2)` | Captures both unigrams and informative bigrams (`mr ebbers`, `australian open`). |
| `min_df` | `3` | Prunes rare words/typos appearing in fewer than 3 documents (~0.14% corpus frequency). |
| `max_df` | `0.7` | Ignores ubiquitous journalistic reporting words appearing in >70% of articles. |
| `max_features` | `5000` | Limits vocabulary to the top 5,000 terms by corpus frequency. |
| `sublinear_tf` | `True` | Applies logarithmic term frequency scaling ($1 + \log(\text{tf})$). |
| `stop_words` | `"english"` | Eliminates 318 generic English function words that carry zero topical significance. |
| `token_pattern` | `(?u)\b[a-zA-Z0-9]+(?:[-'][a-zA-Z0-9]+)*\b` | Preserves intra-word hyphens and contractions. |
| `norm` | `"l2"` | Normalizes document vectors to unit length so Euclidean distance directly reflects cosine similarity. |

---

## Phase 5 — Module 3: K-Means Topic Clustering

Module 3 groups the 2,123 news articles into discoverable topics using K-Means clustering on the sparse TF-IDF feature matrix.

### Key Capabilities

1. `backend/app/ml/clustering.py`:
   - `fit_kmeans(tfidf_matrix, n_clusters=5, random_state=42, n_init=10, max_iter=300, tol=1e-4)`: Fits K-Means directly on sparse CSR matrix without dense array conversion.
   - `predict_cluster_assignments(kmeans_model, tfidf_matrix, article_ids)`: Predicts cluster assignments with strictly verified 1-to-1 mapping back to `article_id`.
   - `get_cluster_centroid_top_terms(kmeans_model, feature_names, top_n=10)`: Extracts the top weighted terms from each cluster centroid in descending weight order.
   - `get_representative_articles(tfidf_matrix, cluster_labels, kmeans_model, article_ids, top_k=3, texts=None)`: Selects representative articles closest to each centroid using sparse Euclidean distance without full matrix densification.
   - `generate_cluster_summaries(...)`: Compiles data-driven topic summaries with neutral integer cluster IDs, cluster sizes, percentages, top terms, and representative articles.
   - `evaluate_clusters_against_labels(...)`: Optional post-hoc evaluation path joining assignments to `evaluation_labels.csv` by `article_id`.
   - `save_kmeans_model(...)` & `load_kmeans_model(...)`: Persists fitted model to `backend/models/kmeans_model.joblib`.
   - `save_cluster_assignments(...)` & `save_cluster_summaries(...)`: Exports assignments to CSV and summaries to JSON.

2. `backend/tests/test_clustering.py`:
   - 12 comprehensive unit and integration tests covering sparse matrix preservation, single assignment per article, article ID alignment, deterministic reproducibility, cluster sizes, centroid top-term sorting, representative article sparse distances, invalid input handling, joblib round-trip serialization, post-hoc label evaluation, strict label isolation, and full corpus integration.

---

## Phase 5 — Module 4: Related-Story Detection

Module 4 retrieves the top-$k$ most similar news stories for any given article in the corpus using cosine similarity over sparse TF-IDF feature vectors.

### How Related-Story Ranking Works

1. **Cosine Similarity Formulation:**
   Because all document vectors in the NewsMap TF-IDF matrix are $L_2$-normalized ($\|\mathbf{x}\|_2 = 1.0$), cosine similarity between source vector $\mathbf{x}_{\text{src}}$ and candidate vector $\mathbf{x}_i$ reduces to their inner dot product:
   $$\text{cosine\_similarity}(\mathbf{x}_{\text{src}}, \mathbf{x}_i) = \frac{\mathbf{x}_{\text{src}} \cdot \mathbf{x}_i}{\|\mathbf{x}_{\text{src}}\|_2 \|\mathbf{x}_i\|_2} = \mathbf{x}_{\text{src}} \cdot \mathbf{x}_i$$

2. **Single-Query Sparse Matrix Multiplication:**
   Rather than materializing a massive $N \times N$ pairwise all-pairs dense matrix (~36 MB, scaling quadratically $O(N^2)$), the service evaluates similarity on demand via a single sparse matrix-vector dot product:
   $$\mathbf{s} = \mathbf{X} \mathbf{x}_{\text{src}}^T$$
   Where $\mathbf{X}$ is the $(2123 \times 5000)$ sparse CSR matrix and $\mathbf{x}_{\text{src}}$ is a $(1 \times 5000)$ sparse vector.
   - **Time Complexity:** Sub-millisecond execution ($< 1$ ms per query).
   - **Space Complexity:** A single 1D vector of length $N$ (2,123 floats = ~17 KB), completely avoiding memory exhaustion.

3. **Exclusion and Ranking Rules:**
   - **Source Exclusion:** The source article itself is strictly masked out from candidate recommendations.
   - **Descending Score Sorting:** Candidates are sorted strictly in descending order of similarity score.
   - **Deterministic Tie-Breaking:** If two candidates share the exact same similarity score, ties are broken deterministically by `article_id` ascending.
   - **Filtering:** Candidates with scores below `min_similarity` or with zero topical overlap ($0.0$) are pruned.
   - **Zero-Feature Handling:** If an article has zero active vocabulary features, an empty list is returned safely without throwing runtime errors.

4. **Cluster-Aware Supplementary Metadata:**
   If Module 3 cluster assignments are available, the service joins `cluster_id` and attaches a `same_cluster` boolean indicator (`True` if the related story belongs to the source article's cluster).
   - *Architectural Note:* Cluster membership is supplementary metadata only. Cosine similarity alone dictates ranking. Same-cluster articles are not automatically related, and related articles may span cluster boundaries. Ground-truth category labels are never accessed.

### Meaning and Limitations of Cosine-Similarity Scores

| Score Range | Typical Relationship | Interpretation |
| :---: | :--- | :--- |
| **0.50 – 1.00** | **Directly Related / Continuing Story** | Articles covering the same unfolding news event, identical named entities, and shared vocabulary (e.g., Bernie Ebbers WorldCom fraud trial). |
| **0.30 – 0.49** | **Closely Related Theme / Sub-Topic** | Articles sharing substantial domain-specific vocabulary and themes (e.g., Oscar contenders, mobile phone network chipsets). |
| **0.15 – 0.29** | **Broader Topic Overlap** | Articles sharing general terminology within an industry or domain (e.g., general economic growth, consumer electronics). |
| **0.00 – 0.14** | **Marginal / Incidental Overlap** | High chance of incidental token sharing rather than narrative relatedness. Filtered out by `min_similarity`. |

**Limitations:**
- **Lexical Vocabulary Dependency:** TF-IDF cosine similarity measures word and bigram co-occurrence. It does not recognize synonyms or paraphrasing absent from the n-gram vocabulary (e.g., "physician" vs. "doctor").
- **Length Normalization:** $L_2$ normalization ensures long documents do not dominate over short ones, but very brief articles with few words may have sparser overlap.

---

## Execution Commands (Windows)

All commands are run from the project root (`D:\NewsMap`) using Windows PowerShell and the existing `.venv`:

### 1. Run Feature Extraction Module (Fits and Saves Vectorizer)
```powershell
Set-Location 'D:\NewsMap'
.\.venv\Scripts\python.exe -m backend.app.ml.feature_extraction
```

### 2. Run Topic Clustering Module (Fits K-Means, Evaluates, and Exports Artifacts)
```powershell
Set-Location 'D:\NewsMap'
.\.venv\Scripts\python.exe -m backend.app.ml.clustering
```

### 3. Run Related-Story Detection Module (Demonstrates Retrieval on Real Corpus)
```powershell
Set-Location 'D:\NewsMap'
.\.venv\Scripts\python.exe -m backend.app.services.related_stories
```

### 4. Run Focused Module 4 Tests Specifically (13 tests)
```powershell
Set-Location 'D:\NewsMap'
.\.venv\Scripts\python.exe -m unittest discover -v -s backend/tests -p "test_related_stories.py"
```

### 5. Run All Project Test Suites (Phase 4 + Phase 5 Modules 1–4 — 68 tests total)
```powershell
Set-Location 'D:\NewsMap'
.\.venv\Scripts\python.exe -m unittest discover -v -s backend/tests -p "test_*.py"
```
