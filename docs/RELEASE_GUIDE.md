# Release & Zenodo DOI Guide

This document describes the workflow for cutting GitHub releases, managing semantic version tags, and minting permanent digital object identifiers (DOIs) via Zenodo.

---

## 1. Release Architecture

Hemolux follows [Semantic Versioning 2.0.0](https://semver.org/):
- `v0.1.0-alpha`: Initial research prototype release containing fairness auditing, zero-build client, and benchmark harness.
- `v0.2.0-beta`: Inclusion of prospective validation results and calibrated offline models.
- `v1.0.0`: Fully validated clinical screening protocol with multicentric IRB clearance.

---

## 2. Release Steps

### Step 1: Pre-Release Verification
Run the complete test suite, linting, and fairness evaluation harness before tagging:

```bash
# Run 900+ unit and integration tests
pytest

# Strict Ruff code-style check
ruff check .

# Validate fairness audit benchmark harness
hemolux eval --predictions sample_preds.csv --format markdown
```

### Step 2: Version Tagging
Tag the git commit with an annotated tag:

```bash
git tag -a v0.1.0-alpha -m "Release v0.1.0-alpha: Research Prototype with Fairness Audit, Selective Abstention, and Benchmark Harness"
git push origin v0.1.0-alpha
```

### Step 3: GitHub Release Creation
1. Navigate to GitHub -> **Releases** -> **Draft a new release**.
2. Select tag `v0.1.0-alpha`.
3. Provide release notes with:
   - Key highlights (e.g. `hemolux eval`, ADR documentation, zero severe cases notice).
   - Artifact checksums (SHA256 for ONNX models).
   - Direct link to the live client: `https://hemolux.pages.dev/`.

---

## 3. Zenodo DOI Minting

1. **GitHub Webhook Integration:**
   - Hemolux repository is linked to Zenodo via GitHub OAuth.
   - Whenever a new GitHub Release is published, Zenodo automatically archives the repository snapshot and assigns a citable DOI (e.g., `10.5281/zenodo.XXXXXX`).
2. **Metadata Synchronization:**
   - Zenodo reads metadata directly from `.zenodo.json` and `CITATION.cff` in the repository root.
3. **Badge Display:**
   - Once minted, update the DOI badge in `README.md`:
   ```markdown
   [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
   ```
