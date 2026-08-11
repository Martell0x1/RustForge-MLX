# RustForge ML — Mathematics Module

The mathematics module is the mathematical foundation of RustForge ML.

The goal is not to build a general-purpose mathematics library. The goal is to implement the mathematical concepts required to understand and build classical machine learning algorithms from scratch.

---

## Philosophy

Every mathematical abstraction should answer:

> What ML concept will this help me understand or implement?

The learning progression is:

```text
Scalar
   ↓
Vector
   ↓
Matrix
   ↓
Statistics
   ↓
Probability
   ↓
Distributions
   ↓
Decomposition
   ↓
Optimization
   ↓
Classical ML
---

1. Scalar

A scalar is represented directly using Rust's numeric types.

For ML, the primary type will be:

f64

Useful operations:

abs()
sqrt()
powi()
powf()
exp()
ln()
sin()
cos()

No custom Scalar type is required initially.

2. Vector

Vector is the first major mathematical type.

Possible representation:

pub struct Vector {
    data: Vec<f64>,
}
Construction
new()
zeros()
ones()
from_slice()
Properties
len()
is_empty()
Access
get()
set()
Element-wise operations
add()
subtract()
multiply()
divide()
Scalar operations
scale()
Linear algebra
dot()
norm()
normalize()
distance()
Aggregation
sum()
mean()
min()
max()
argmin()
argmax()
Utility operations
map()
zip()

3. Matrix

Matrix will be one of the most important types in RustForge ML.

Possible representation:

pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

Prefer a flat Vec<f64> rather than Vec<Vec<f64>>.

Example:

[ 1 2 3 ]
[ 4 5 6 ]

can be stored as:

[1, 2, 3, 4, 5, 6]
Construction
new()
zeros()
ones()
identity()
from_rows()
from_flat()
Dimensions
rows()
cols()
shape()
Access
get()
set()
row()
column()
Basic arithmetic
add()
subtract()
scale()
Matrix operations
transpose()
multiply()

Example:

A × B
Matrix-vector operations
mat_vec_mul()

Example:

A × x

This operation will be heavily used by ML algorithms.

Linear algebra

Eventually implement:

determinant()
inverse()
trace()
rank()
solve()

Especially:

Ax = b
4. Statistics

Statistics will primarily consist of mathematical functions.

Central tendency
mean()
median()
mode()
Dispersion
variance()
std_dev()
range()
Relationships
covariance()
correlation()

These are particularly important for machine learning.

Quantiles
percentile()
quartile()
Aggregation
sum()
min()
max()
5. Probability

Create:

probability.rs

Important concepts:

probability()
conditional_probability()
joint_probability()
marginal_probability()
bayes()

Important relationship:

P(A | B)

Bayes' theorem:

P(A | B) = P(B | A) P(A) / P(B)

This will later be used to implement Naive Bayes.

6. Probability Distributions

Eventually create:

distributions/
├── normal.rs
├── bernoulli.rs
├── binomial.rs
└── uniform.rs

Each distribution can eventually provide:

pdf()
pmf()
cdf()
mean()
variance()
sample()

For example:

Normal {
    mean: f64,
    std_dev: f64,
}

with operations such as:

pdf(x)
cdf(x)
sample()

This module is not an immediate priority.

7. Matrix Decompositions

Eventually create:

decomposition/
├── lu.rs
├── qr.rs
├── eigen.rs
└── svd.rs
LU Decomposition

Used for:

solving linear systems
matrix inversion
numerical linear algebra
QR Decomposition

Used for:

solving linear systems
numerical stability
least squares
Eigenvalues / Eigenvectors

Implement:

eigenvalues()
eigenvectors()

These are critical for understanding PCA.

Singular Value Decomposition

Implement:

SVD

Useful for:

PCA
dimensionality reduction
recommender systems
numerical linear algebra

These should be implemented much later.

8. Optimization

Eventually create:

optimization/
├── gradient_descent.rs
├── sgd.rs
└── ...

Implement:

gradient_descent()
sgd()
mini_batch_gradient_descent()

Important concepts:

gradient
learning rate
convergence
iterations
optimization objective

Later:

momentum
Adam

Adam is more important once neural networks are introduced.

9. Loss Functions

Loss functions are technically mathematical foundations rather than general-purpose ML models.

Eventually create:

loss/
├── mse.rs
├── mae.rs
├── log_loss.rs
└── hinge.rs

Implement:

mse()
mae()
binary_cross_entropy()
hinge_loss()

These will later be used by the ML algorithms.

Recommended Module Structure

Eventually:

src/
└── math/
    ├── mod.rs
    │
    ├── vector.rs
    ├── matrix.rs
    ├── statistics.rs
    ├── probability.rs
    │
    ├── distributions/
    │   ├── mod.rs
    │   ├── normal.rs
    │   ├── bernoulli.rs
    │   ├── binomial.rs
    │   └── uniform.rs
    │
    ├── decomposition/
    │   ├── mod.rs
    │   ├── lu.rs
    │   ├── qr.rs
    │   ├── eigen.rs
    │   └── svd.rs
    │
    └── optimization/
        ├── mod.rs
        ├── gradient_descent.rs
        └── sgd.rs

Loss functions can later become a separate top-level module:

src/
└── loss/
    ├── mod.rs
    ├── mse.rs
    ├── mae.rs
    ├── log_loss.rs
    └── hinge.rs