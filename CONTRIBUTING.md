# Contributing to LOWKEY

Thank you for your interest in contributing to **LOWKEY**! We are committed to building an open, robust, and mathematically grounded music recommendation engine.

This document outlines the engineering guidelines, development workflow, and pull request standards required for all contributors.

---

## Code of Conduct

All contributors and maintainers are expected to adhere to our [Code of Conduct](CODE_OF_CONDUCT.md). Please report any unacceptable behavior to the maintainers.

---

## Development Workflow

### 1. Fork & Branching Strategy
We adhere to a standard Trunk-Based / Feature-Branch development model:
* `main` is our production-ready, protected branch. All code in `main` must pass CI and all verification tests.
* When working on an issue or improvement, create a descriptive feature branch from `main`:
  ```bash
  git checkout -b feature/audio-vector-hybrid-scoring
  # or
  git checkout -b fix/chroma-metadata-parsing
  ```

### 2. Environment Setup
Clone your fork and install dependencies:
```bash
git clone https://github.com/ritigyam2/lowkey.git
cd lowkey/backend
python -m pip install -r requirements.txt
```

Recommended development packages for linting and formatting:
```bash
python -m pip install black flake8 mypy pytest
```

---

## Commit Message Conventions

We strictly follow the **[Conventional Commits](https://www.conventionalcommits.org/)** specification. All commit messages must follow this structure:

```
<type>(<scope>): <short description in imperative mood>

[optional body explaining the motivation and context]

[optional footer(s), e.g. Fixes #123]
```

### Allowed Types:
* `feat`: A new feature or algorithm (e.g., `feat(retrieval): add hybrid scoring ranker`)
* `fix`: A bug fix (e.g., `fix(hnsw): clamp tempo outlier values in scaler`)
* `docs`: Documentation updates only (e.g., `docs(readme): add mathematical formulation`)
* `style`: Code style / formatting changes (no functional changes)
* `refactor`: Code change that neither fixes a bug nor adds a feature
* `perf`: Performance optimizations (e.g., `perf(ingest): optimize batch memory footprint`)
* `test`: Adding or correcting tests
* `chore`: Build system, dependency, or configuration updates

---

## Engineering & Code Standards

1. **Strict Type Annotations:** All new Python functions, methods, and classes must include PEP 484 type annotations.
2. **Deterministic Preprocessing:** Audio features, scaling formulas, and normalization pipelines must be deterministic and reproducible.
3. **Low-Memory Footprint:** Ensure all ingestion, batching, and vector generation pipelines maintain a low memory ceiling ($< 150\,\text{MB}$ RAM), utilizing explicit streaming and garbage collection where necessary.
4. **Code Formatting:** Code must pass `black` with standard line length limits (88–100 characters).

---

## Testing & Verification

Before opening a pull request, run the test harnesses to ensure zero regressions:

```bash
# Verify HNSW Audio Similarity Engine
python backend/test_audio_hnsw.py

# Verify ChromaDB Semantic Engine (if applicable)
python backend/verify_lowkey.py
```

All test assertions must pass before submitting your PR.

---

## Pull Request Submission Checklist

When opening a Pull Request:
1. Provide a clear and descriptive title following the Conventional Commits format.
2. Fill out the [Pull Request Template](.github/pull_request_template.md).
3. Confirm all existing and new tests pass locally.
4. Ensure documentation is updated if your changes impact the API, dataset schemas, or indexing engines.
5. Keep PRs focused on a single logical change to facilitate efficient review.
