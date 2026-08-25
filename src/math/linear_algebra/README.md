# rustMLX — Linear Algebra Module Structure

```text
src/
└── math/
    ├── mod.rs
    ├── vector.rs
    ├── matrix.rs
    │
    └── linear_algebra/
        ├── mod.rs
        │
        ├── operations.rs
        ├── decomposition.rs
        ├── solving.rs
        ├── determinant.rs
        ├── inverse.rs
        ├── eigen.rs
        ├── norms.rs
        ├── orthogonal.rs
        ├── least_squares.rs
        ├── svd.rs
        └── properties.rs
```

## 1. `operations.rs`

Fundamental linear-algebra operations that operate on `Matrix` and `Vector`.

```text
matrix_vector_mul
vector_matrix_mul
dot_product
outer_product
cross_product
hadamard_product
```

Keep basic element-wise operations in `Matrix` itself; this module should contain operations that conceptually combine linear-algebra objects.

---

## 2. `properties.rs`

Operations that determine structural properties of matrices.

```text
is_square
is_symmetric
is_skew_symmetric
is_diagonal
is_identity
is_upper_triangular
is_lower_triangular
is_singular
rank
```

Potential future additions:

```text
is_positive_definite
is_positive_semidefinite
```

---

## 3. `solving.rs`

Solving linear systems.

```text
solve
solve_gaussian
solve_lu
solve_cholesky
```

Primary problem:

```text
Ax = b
```

This module should eventually expose a high-level:

```text
solve(A, b)
```

while internally selecting/using an appropriate algorithm.

---

## 4. `decomposition.rs`

Matrix decompositions.

```text
LU decomposition
QR decomposition
Cholesky decomposition
```

Represent the results explicitly.

For example:

```text
LU
├── L
├── U
└── P
```

and:

```text
QR
├── Q
└── R
```

This is an important module because `determinant`, `inverse`, `solve`, least squares, and eigenvalue algorithms can reuse these decompositions.

---

## 5. `determinant.rs`

Determinant-related functionality.

```text
determinant
log_determinant
```

Internally, don't build a separate algorithm for every matrix operation.

Prefer:

```text
det(A)
   │
   └── LU decomposition
          │
          └── determinant
```

For numerical work, this is much more appropriate than recursive cofactor expansion.

---

## 6. `inverse.rs`

Matrix inversion.

```text
inverse
```

Possible implementations:

```text
inverse_gauss_jordan
inverse_lu
```

Eventually the public API should simply be:

```text
inverse(A)
```

and the implementation can use LU or another appropriate algorithm.

---

## 7. `norms.rs`

Vector and matrix norms.

You already have Frobenius norm in `Matrix`, but the linear-algebra module should eventually cover the standard numerical norms.

### Vector

```text
L1
L2
L∞
```

### Matrix

```text
Frobenius
L1
L∞
spectral norm
```

Also eventually:

```text
condition_number
```

---

## 8. `orthogonal.rs`

Orthogonality and orthogonalization.

```text
projection
gram_schmidt
modified_gram_schmidt
orthonormalize
```

This module leads directly into QR decomposition and least-squares problems.

Conceptually:

```text
Vectors
   │
   └── Orthogonalization
          │
          └── QR
```

---

## 9. `least_squares.rs`

Least-squares problems.

```text
least_squares
normal_equation
```

Problem:

```text
Ax ≈ b
```

For a serious numerical implementation, prefer solving through QR rather than blindly calculating:

```text
(AᵀA)⁻¹Aᵀb
```

because explicitly forming the inverse is numerically inferior.

Later this module can support:

```text
weighted_least_squares
```

---

## 10. `eigen.rs`

Eigenvalue/eigenvector algorithms.

```text
eigenvalues
eigenvectors
power_iteration
```

Later:

```text
qr_algorithm
symmetric_eigen
```

Core equation:

```text
Av = λv
```

This module should eventually support both:

```text
general matrices
symmetric matrices
```

with specialized algorithms where appropriate.

---

## 11. `svd.rs`

Singular Value Decomposition.

```text
svd
```

Produces:

```text
A = UΣVᵀ
```

Eventually:

```text
thin_svd
full_svd
```

This module is important for ML because it enables:

```text
PCA
dimensionality reduction
pseudoinverse
low-rank approximation
least squares
```

---

# Recommended dependency graph

The important part isn't just the files; it's how they depend on each other.

```text
                    Matrix
                      │
                    Vector
                      │
                      ▼
                operations.rs
                      │
          ┌───────────┼────────────┐
          ▼           ▼            ▼
     properties     norms       orthogonal
          │                        │
          ▼                        ▼
     decomposition ◄────────────── QR
          │
      ┌───┼────┐
      ▼   ▼    ▼
     LU  QR  Cholesky
      │   │     │
      │   │     │
      ▼   ▼     ▼
  solving  least_squares
      │
      ▼
  determinant
      │
      ▼
   inverse

eigen
  │
  └── QR algorithms

svd
  │
  ├── pseudoinverse
  ├── PCA
  └── low-rank approximation
```

# What I would NOT put in `linear_algebra`

Don't turn it into a dumping ground.

These belong elsewhere:

```text
Activation functions       → neural_network / nn
Loss functions             → loss
Gradient descent            → optimization
Random initialization      → random / initialization
Convolution                → nn / convolution
Tensor broadcasting        → tensor
Autograd                    → autograd
Statistics                  → statistics
```

The responsibility should be:

```text
linear_algebra
    ↓
"How do I manipulate and solve mathematical linear systems?"
```

not:

```text
linear_algebra
    ↓
"Everything related to ML mathematics"
```

# Final target

Eventually your math layer should look more like:

```text
math/
│
├── vector.rs
├── matrix.rs
├── tensor.rs
│
├── linear_algebra/
│   ├── operations.rs
│   ├── properties.rs
│   ├── norms.rs
│   ├── solving.rs
│   ├── decomposition.rs
│   ├── determinant.rs
│   ├── inverse.rs
│   ├── orthogonal.rs
│   ├── least_squares.rs
│   ├── eigen.rs
│   └── svd.rs
│
├── statistics/
├── probability/
├── optimization/
└── calculus/
```

For the **next implementation step**, I would create only these four files first:

```text
linear_algebra/
├── mod.rs
├── operations.rs
├── properties.rs
└── solving.rs
```

Then implement the remaining modules as the algorithms actually require them. This keeps the architecture real without creating a dozen empty modules.

