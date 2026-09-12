# RustForge-MLX

**A modular machine-learning toolkit for Rust — from mathematics and data processing to classical ML and, eventually, neural networks.**

RustForge-MLX is an open-source machine-learning library written entirely in Rust. The project is designed to provide a practical, composable foundation for understanding and building machine-learning systems without hiding the underlying mathematics and algorithms behind opaque abstractions.

The goal is to grow from a strong mathematical and data-processing foundation into a broader machine-learning toolkit covering classical ML and eventually neural-network functionality.

> **Status:** Early development
> **License:** MIT
> **Language:** Rust

---

## Overview

RustForge-MLX is organized as a layered machine-learning ecosystem:

```text
                              RustForge-MLX
                                    │
                 ┌──────────────────┴──────────────────┐
                 │                                     │
           Mathematics                            Data System
                 │                                     │
       ┌─────────┼─────────┐                    ┌──────┼──────┐
       │         │         │                    │      │      │
 Linear Algebra Probability Statistics       Dataset  CSV  Preprocessing
       │         │         │                    │      │      │
       └─────────┴─────────┴────────────────────┴──────┴──────┘
                                    │
                                    ▼
                            Classical ML
                                    │
             ┌──────────────────────┼──────────────────────┐
             │                      │                      │
        Regression            Classification          Clustering
             │                      │                      │
      ┌──────┼──────┐       ┌───────┼───────┐       ┌─────┼─────┐
      │      │      │       │       │       │       │     │     │
    Linear  Ridge  Lasso  Logistic   KNN   SVM    K-Means DBSCAN GMM
    Poly.                 Naive Bayes
                          Decision Tree
                          Random Forest
                                    │
                                    ▼
                            Model Evaluation
                                    │
                       ┌────────────┼────────────┐
                       │            │            │
                    Metrics      Cross-Val.   Tuning
                       │            │            │
                Accuracy/F1      K-Fold      Grid Search
                Precision        Stratified  Random Search
                Recall
                ROC-AUC
                                    │
                                    ▼
                            Neural Networks
                               (Future)
```

The architecture is intentionally layered: higher-level machine-learning algorithms should be able to build on reusable mathematical and data-processing primitives.

---

## Why RustForge-MLX?

Machine learning libraries are often consumed entirely through high-level APIs. That is convenient, but it can make it difficult to understand what is actually happening underneath.

RustForge-MLX takes a different approach:

* **Rust-first** — leverage Rust's safety, performance, and expressive type system.
* **Modular** — separate mathematics, data processing, algorithms, and evaluation.
* **Practical** — focus on usable implementations rather than purely academic exercises.
* **Transparent** — algorithms should remain understandable and inspectable.
* **Composable** — reusable primitives should serve as building blocks for higher-level models.
* **Tested** — numerical algorithms should be validated with meaningful tests and tolerances.
* **Minimal** — avoid unnecessary dependencies and abstractions.

The project is also intended to be a place for experimenting with how machine-learning primitives can be designed in idiomatic Rust.

---

## Current Scope

RustForge-MLX is being developed incrementally.

### Mathematics

The mathematical foundation is the first layer of the project.

Planned and developing areas include:

* Linear algebra
* Vectors and matrices
* Matrix operations
* Matrix decompositions
* Probability
* Statistics
* Numerical methods
* Optimization

### Data System

The data layer provides the primitives required to prepare datasets for machine-learning algorithms.

Planned areas include:

* Dataset abstractions
* CSV loading
* Train/test splitting
* Normalization
* Feature preprocessing
* Dataset transformations

### Classical Machine Learning

The classical ML layer builds algorithms on top of the mathematical and data foundations.

#### Regression

* Linear Regression
* Polynomial Regression
* Ridge Regression
* Lasso Regression

#### Classification

* Logistic Regression
* K-Nearest Neighbors
* Naive Bayes
* Decision Trees
* Random Forests
* Support Vector Machines

#### Clustering

* K-Means
* DBSCAN
* Gaussian Mixture Models

### Model Evaluation

Evaluation utilities will provide reusable tools for measuring and comparing models.

Planned areas include:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Cross-validation
* K-Fold validation
* Stratified validation
* Hyperparameter tuning
* Grid search
* Random search

### Neural Networks

Neural-network functionality is a longer-term goal.

The current priority is to establish a reliable mathematical, numerical, and data-processing foundation before building higher-level neural-network abstractions.

---

## Quick Start

### Requirements

You need:

* Rust stable
* Cargo
* Git

Verify your installation:

```bash
rustc --version
cargo --version
```

### Clone

```bash
git clone https://github.com/Martell0x1/RustForge-MLX.git
cd RustForge-MLX
```

### Build

```bash
cargo build
```

### Run Tests

```bash
cargo test
```

### Format

```bash
cargo fmt
```

### Check Formatting

```bash
cargo fmt --check
```

### Run Clippy

```bash
cargo clippy --all-targets --all-features -- -D warnings
```

---

## Example

As the library develops, usage should remain simple and composable.

For example, a future workflow may look like:

```rust
use rustforge_mlx::linear_algebra::Matrix;

fn main() {
    let matrix = Matrix::from_vec(
        2,
        2,
        vec![
            1.0, 2.0,
            3.0, 4.0,
        ],
    );

    let transposed = matrix.transpose();

    println!("{transposed:?}");
}
```

The exact public API is still evolving, so examples in the repository should be considered the authoritative source for currently supported functionality.

---

## Design Principles

RustForge-MLX follows several principles when introducing new functionality.

### 1. Build from fundamentals

Higher-level algorithms should reuse lower-level primitives where practical.

For example:

```text
Matrix
  ↓
Matrix Operations
  ↓
Numerical Methods
  ↓
ML Algorithms
  ↓
Model Evaluation
```

This keeps the architecture understandable and reduces duplicated implementations.

### 2. Prefer idiomatic Rust

The project favors:

* Clear ownership and borrowing
* Strong type safety
* Explicit error handling
* Small and composable APIs
* Minimal unnecessary abstraction
* Standard Rust conventions

### 3. Correctness before optimization

Machine-learning algorithms can produce plausible-looking results even when their implementations are mathematically incorrect.

For numerical code:

* Test normal cases.
* Test boundary conditions.
* Test invalid inputs.
* Test numerical edge cases.
* Use appropriate floating-point tolerances.
* Add regression tests for discovered bugs.

For example:

```rust
assert!((result - expected).abs() < 1e-10);
```

Performance improvements should preserve correctness.

### 4. Keep dependencies intentional

A dependency should provide meaningful value to the project.

Before introducing a new dependency, consider whether the functionality can reasonably be implemented using the existing standard library or project primitives.

### 5. Keep APIs predictable

Similar operations should behave consistently across the library.

An API should make the mathematical or machine-learning operation recognizable rather than introducing abstractions that obscure it.

---

## Project Structure

The project is expected to grow around several logical layers:

```text
RustForge-MLX/
├── src/
│   ├── math/
│   │   ├── linear_algebra/
│   │   ├── probability/
│   │   ├── statistics/
│   │   └── optimization/
│   │
│   ├── data/
│   │   ├── dataset/
│   │   ├── csv/
│   │   └── preprocessing/
│   │
│   ├── ml/
│   │   ├── regression/
│   │   ├── classification/
│   │   └── clustering/
│   │
│   └── evaluation/
│
├── tests/
├── examples/
├── benches/
├── docs/
├── Cargo.toml
└── LICENSE
```

The exact structure may evolve as the architecture develops.

---

## Roadmap

RustForge-MLX is intentionally being built in stages.

### Phase 1 — Mathematical Foundation

* [ ] Vector operations
* [ ] Matrix operations
* [ ] Matrix decompositions
* [ ] Numerical methods
* [ ] Probability primitives
* [ ] Statistical utilities
* [ ] Optimization primitives

### Phase 2 — Data System

* [ ] Dataset abstraction
* [ ] CSV loading
* [ ] Train/test splitting
* [ ] Normalization
* [ ] Feature preprocessing
* [ ] Dataset transformations

### Phase 3 — Classical ML

* [ ] Linear Regression
* [ ] Polynomial Regression
* [ ] Ridge Regression
* [ ] Lasso Regression
* [ ] Logistic Regression
* [ ] KNN
* [ ] Naive Bayes
* [ ] Decision Trees
* [ ] Random Forest
* [ ] SVM
* [ ] K-Means
* [ ] DBSCAN
* [ ] GMM

### Phase 4 — Evaluation

* [ ] Classification metrics
* [ ] Regression metrics
* [ ] Cross-validation
* [ ] K-Fold validation
* [ ] Stratified validation
* [ ] Hyperparameter tuning

### Phase 5 — Neural Networks

* [ ] Tensor abstractions
* [ ] Automatic differentiation
* [ ] Layers
* [ ] Activations
* [ ] Loss functions
* [ ] Optimizers
* [ ] Training utilities

This roadmap is a direction rather than a fixed promise. APIs and implementation priorities may change as the project evolves.

---

## Contributing

Contributions are welcome.

You can contribute through:

* New algorithms
* Mathematical primitives
* Bug fixes
* Tests
* Documentation
* Examples
* Benchmarks
* Performance improvements
* API improvements
* Feature proposals
* Numerical correctness improvements

### Before Starting Large Changes

For large features or architectural changes, open an issue first.

This allows the design and API to be discussed before significant implementation work begins.

For small fixes, tests, documentation changes, and minor improvements, a pull request can usually be opened directly.

### Development Workflow

Create a dedicated branch:

```bash
git checkout -b feature/matrix-inverse
```

Examples:

```text
feature/matrix-inverse
feature/kmeans
feature/decision-tree
fix/gaussian-elimination
docs/linear-algebra
test/vector-operations
```

Before opening a pull request:

```bash
cargo fmt
cargo test
cargo clippy --all-targets --all-features -- -D warnings
```

### Pull Requests

Keep pull requests focused on one logical change whenever possible.

A good pull request should explain:

* What changed
* Why the change was needed
* How it was implemented
* Tests added or updated
* Performance considerations
* Any API changes
* Any breaking changes

Example titles:

```text
feat: add QR decomposition
fix: handle singular matrices in Gaussian elimination
test: add SVD decomposition tests
docs: document vector API
refactor: simplify matrix multiplication
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the complete contribution guidelines.

---

## Documentation

Public APIs should be documented where appropriate.

When introducing a mathematical concept or significant module, consider adding documentation under:

```text
docs/
```

Good documentation should explain both:

1. **What** the API does.
2. **Why/how** the underlying mathematical or technical concept works.

Documentation-only contributions are welcome.

---

## Bug Reports

When reporting a bug, please include:

* A clear description of the problem
* Steps to reproduce it
* Expected behavior
* Actual behavior
* Rust version
* Relevant code
* A minimal reproduction when possible

For numerical issues, include the input values and expected numerical behavior whenever possible.

---

## Feature Requests

Feature requests are welcome.

For larger proposals, please describe:

* The problem being solved
* Why the feature belongs in RustForge-MLX
* Proposed API or design
* Possible alternatives
* Relevant mathematical or technical references
* Potential compatibility or performance implications

---

## Where to Start

If you're new to the project, good places to start are:

* Documentation improvements
* Tests for existing functionality
* Examples
* Small mathematical utilities
* Bug fixes
* Benchmarks
* Issues labeled `good first issue`
* Issues labeled `help wanted`

If you're unsure about an implementation, open an issue and discuss the approach before starting a substantial change.

---

## Community

RustForge-MLX aims to maintain a welcoming and constructive open-source community.

Please:

* Be respectful.
* Assume good intentions.
* Give constructive feedback.
* Focus criticism on ideas and implementations rather than people.
* Help other contributors learn.
* Keep technical discussions focused and professional.

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for the project's full community standards.

---

## License

RustForge-MLX is distributed under the MIT License.

See [LICENSE](LICENSE) for the complete license text.

---

## Project Status

RustForge-MLX is an evolving open-source project.

The API, module structure, and roadmap may change as the mathematical and machine-learning foundations mature.

Contributions, experimentation, and constructive discussion are encouraged.

**Build the foundations. Understand the algorithms. Make machine learning Rust-native.**
