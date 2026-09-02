# Contributing to RustForge-MLX

Thank you for your interest in contributing to RustForge-MLX!

RustForge-MLX is an open-source machine learning toolkit written in Rust. The project aims to provide a practical, modular foundation for machine learning, mathematics, data processing, classical ML, and eventually neural networks.

Contributions of all kinds are welcome — code, documentation, tests, examples, bug reports, performance improvements, and ideas.

## Before You Start

For large features or architectural changes, please open an issue first so we can discuss the design before implementation.

For small bug fixes, documentation improvements, tests, or minor improvements, you can usually open a pull request directly.

## Development Setup

### Requirements

* Rust stable toolchain
* Cargo
* Git

Verify your Rust installation:

```bash
rustc --version
cargo --version
```

Clone the repository:

```bash
git clone https://github.com/Martell0x1/RustForge-MLX.git
cd RustForge-MLX
```

Build the project:

```bash
cargo build
```

Run the test suite:

```bash
cargo test
```

Format the code:

```bash
cargo fmt
```

Check formatting:

```bash
cargo fmt --check
```

Run Clippy:

```bash
cargo clippy --all-targets --all-features -- -D warnings
```

## Branches

Please create a dedicated branch for your contribution rather than working directly on `main`.

Examples:

```text
feature/matrix-inverse
feature/kmeans
fix/gaussian-elimination
docs/linear-algebra
test/vector-operations
```

## Code Guidelines

RustForge-MLX follows standard Rust conventions.

Please:

* Run `cargo fmt` before submitting a pull request.
* Run the test suite with `cargo test`.
* Run Clippy when appropriate.
* Keep APIs consistent with the existing project design.
* Prefer clear and idiomatic Rust over unnecessary abstraction.
* Avoid introducing dependencies unless they provide meaningful value.
* Add tests for new functionality and bug fixes.
* Document public APIs appropriately.

## Mathematical Implementations

For mathematical algorithms, correctness is especially important.

When adding or modifying an algorithm:

1. Explain the algorithm clearly in the implementation or accompanying documentation.
2. Include tests covering normal cases.
3. Include edge cases where appropriate.
4. Verify numerical behavior with suitable tolerances.
5. Avoid silently returning incorrect results for invalid input.

For algorithms involving floating-point calculations, use appropriate numerical tolerances rather than relying on exact equality.

Example:

```rust
assert!((result - expected).abs() < 1e-10);
```

## Tests

Every new feature should include tests where practical.

Tests should cover:

* Normal input
* Boundary conditions
* Invalid input
* Numerical edge cases
* Regression cases for fixed bugs

Run all tests before submitting:

```bash
cargo test
```

## Pull Requests

Please keep pull requests focused on one change whenever possible.

A good pull request should include:

* A clear title
* A description of what changed
* Why the change was needed
* Tests added or updated
* Any relevant performance considerations
* Any breaking API changes

For example:

```text
feat: add QR decomposition
```

or:

```text
fix: handle singular matrices in Gaussian elimination
```

## Commit Messages

Use clear and descriptive commit messages.

Preferred examples:

```text
feat: add matrix transpose operation
fix: handle singular matrices in LU decomposition
test: add SVD decomposition tests
docs: document vector API
refactor: simplify matrix multiplication
```

## Documentation

Public APIs should be documented where appropriate.

If you introduce a new module or mathematical concept, consider adding documentation under `docs/` as well.

Documentation contributions are welcome even if you are not modifying code.

## Reporting Bugs

When reporting a bug, please provide:

* A description of the problem
* Steps to reproduce it
* Expected behavior
* Actual behavior
* Relevant Rust version
* Relevant code or a minimal reproduction

## Feature Requests

Feature requests are welcome.

For larger features, please explain:

* The problem the feature solves
* The proposed API or design
* Possible alternatives
* Any relevant mathematical or technical references

## Areas Where Contributions Are Welcome

Some areas of the project that may benefit from contributions include:

* Linear algebra
* Matrix decompositions
* Numerical methods
* Probability and statistics
* Optimization
* Dataset utilities
* Classical machine learning
* Model evaluation
* Performance optimization
* Documentation
* Testing
* Examples and benchmarks
* Eventually, neural-network functionality

If you are unsure where to start, look for issues labeled:

* `good first issue`
* `help wanted`

or open an issue and discuss the idea with the maintainers.

## Code of Conduct

Please be respectful and constructive when interacting with other contributors.

Harassment, discrimination, personal attacks, and intentionally disruptive behavior are not welcome.

## License

By contributing to RustForge-MLX, you agree that your contributions will be licensed under the same license as the project.

RustForge-MLX is distributed under the MIT License. See [`LICENSE`](LICENSE) for details.

## Questions

If you are unsure about an implementation, architecture decision, or contribution, feel free to open an issue for discussion before starting substantial work.

Thank you for helping build RustForge-MLX!
