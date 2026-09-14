# Getting Started with RustForge-MLX

Welcome to **RustForge-MLX**! This guide will take you step-by-step from installing the required prerequisites to running your first RustForge-MLX code and preparing your first contribution.

Whether you are new to machine learning in Rust or an experienced developer, this guide will help you get up and running quickly.

---

## 1. Prerequisites

To build and contribute to RustForge-MLX, you will need:

* **Rust stable toolchain** (Rust 2024 edition or later)
* **Cargo** (Rust's package manager and build system, bundled with Rust)
* **Git** (for version control)

If you haven't installed Rust yet, you can install it using [rustup](https://rustup.rs/):

```bash
# On Linux / macOS
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# On Windows, download and run rustup-init.exe from https://rustup.rs/
```

### Verifying Your Installation

Open your terminal or command prompt and verify that all tools are accessible:

```bash
rustc --version
cargo --version
git --version
```

You should see output displaying the installed versions for each tool.

---

## 2. Clone the Repository

Clone the RustForge-MLX repository from GitHub and navigate into the project root:

```bash
git clone https://github.com/Martell0x1/RustForge-MLX.git
cd RustForge-MLX
```

---

## 3. Build the Project

Compile the project and download any internal dependencies using Cargo:

```bash
cargo build
```

This compiles both the library crate (`rustforge_mlx`) and the binary target (`src/main.rs`). A successful build confirms your toolchain is configured correctly.

---

## 4. Run the Test Suite

Before writing new code or making modifications, run the existing test suite:

```bash
cargo test
```

Running tests ensures that everything currently works as expected on your system and provides a clean baseline before you make any changes.

---

## 5. Format and Lint

RustForge-MLX follows standard Rust formatting and linting conventions. Two essential tools are used:

### Formatting (`cargo fmt`)

To automatically format all Rust source files according to the standard style:

```bash
cargo fmt
```

To verify if the codebase is already formatted properly without modifying files (used in CI):

```bash
cargo fmt --check
```

### Linting (`cargo clippy`)

Clippy catches common mistakes and provides idiomatic suggestions. Run Clippy across all targets:

```bash
cargo clippy --all-targets --all-features -- -D warnings
```

* The `-D warnings` flag treats any compiler warnings or clippy suggestions as errors, ensuring clean, high-quality code.

---

## 6. First RustForge-MLX Example

Here is a quick example demonstrating how to use the current mathematical primitives available in `rustforge_mlx`:

```rust
use rustforge_mlx::math::matrix::Matrix;
use rustforge_mlx::math::vector::Vector;

fn main() {
    println!("=== RustForge-MLX Quickstart ===");

    // 1. Vector Operations
    let v1 = Vector::new(vec![1.0, 2.0, 3.0]);
    let v2 = Vector::new(vec![4.0, 5.0, 6.0]);

    // Vector addition (element-wise)
    let v_sum = v1 + v2;
    println!("Vector sum: {:?}", v_sum);

    // 2. Matrix Creation and Manipulation
    let m = Matrix::new(vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
    ]);

    println!("Matrix shape: {:?}", m.shape()); // (rows: 2, cols: 3)

    // Transpose the matrix
    let m_transposed = m.transpose();
    println!("Transposed shape: {:?}", m_transposed.shape()); // (rows: 3, cols: 2)
    println!("Transposed matrix: {:?}", m_transposed);
}
```

You can test this snippet by placing it in `src/main.rs` and running:

```bash
cargo run
```

For more in-depth mathematical documentation, see the [Mathematics Module Documentation](math-module.md).

---

## 7. Repository Structure

Here is an overview of the key directories and files in RustForge-MLX:

```text
RustForge-MLX/
├── Cargo.toml          # Project manifest, package metadata, and dependencies
├── Cargo.lock          # Dependency version lockfile
├── LICENSE             # MIT License terms
├── CONTRIBUTING.md     # Detailed contribution rules and guidelines
├── CODE_OF_CONDUCT.md  # Community standards
├── README.md           # High-level architecture, roadmap, and overview
├── docs/               # In-depth module guides and documentation
│   ├── getting-started.md  # This getting-started guide
│   └── math-module.md      # Mathematics module architecture and concepts
├── src/                # Library and binary source code
│   ├── lib.rs          # Main library entry point (`pub mod math;`)
│   ├── main.rs         # Standalone binary entry point
│   └── math/           # Mathematics foundational layer
│       ├── mod.rs      # Math module exports
│       ├── vector.rs   # Generic Vector<T> implementation and operators
│       ├── matrix.rs   # Generic Matrix<T> implementation and operators
│       └── linear_algebra/ # Linear algebra operations and algorithms
└── tests/              # Integration and end-to-end test suites
```

---

## 8. Making a Contribution

Ready to contribute? Follow this brief workflow:

1. **Create a feature branch**:
   Always create a dedicated branch for your changes instead of committing directly to `main`:
   ```bash
   git checkout -b feature/my-change
   ```

2. **Implement your changes**:
   Write clean, idiomatic Rust code and add tests for any new functionality or bug fixes.

3. **Verify locally**:
   Ensure all checks pass before opening a pull request:
   ```bash
   cargo fmt
   cargo test
   cargo clippy --all-targets --all-features -- -D warnings
   ```

4. **Commit with clear messages**:
   Use descriptive Conventional Commits (e.g., `feat: add matrix determinant`, `docs: update getting-started guide`):
   ```bash
   git add .
   git commit -m "feat: add matrix determinant"
   ```

5. **Open a Pull Request**:
   Push your branch to your GitHub fork and open a Pull Request against `main`.

For complete details on coding style, mathematical tolerances, and PR requirements, please read [`CONTRIBUTING.md`](../CONTRIBUTING.md).
