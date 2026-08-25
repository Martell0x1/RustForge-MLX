//! Linear-system solvers for `Ax = b`.
//!
//! This module implements classical dense algorithms:
//!
//! - Gaussian elimination with partial pivoting
//! - Gauss–Jordan elimination with partial pivoting
//! - Doolittle LU factorization with partial pivoting (`PA = LU`)
//! - Householder QR factorization (`A = QR`)
//! - Cholesky factorization (`A = L Lᵀ`)
//!
//! All solvers operate on the existing [`Matrix`](crate::math::matrix::Matrix)
//! and [`Vector`](crate::math::vector::Vector) types and return
//! [`LinearAlgebraError`] for invalid input or numerically singular systems.

mod cholesky;
mod gauss_jordan;
mod gaussian;
mod lu;
mod qr;

pub use cholesky::{CholeskyDecomposition, cholesky_decompose, cholesky_solve};
pub use gauss_jordan::gauss_jordan_solve;
pub use gaussian::gaussian_solve;
pub use lu::{LUDecomposition, lu_decompose, lu_solve};
pub use qr::{QRDecomposition, qr_decompose, qr_solve};

use crate::math::matrix::Matrix;
use crate::math::vector::Vector;
use std::fmt;

/// Absolute tolerance used to decide whether a pivot is numerically zero.
///
/// A value `x` is treated as zero when `x.abs() < EPSILON`. This is an
/// absolute threshold, so extremely scaled systems may need a different
/// criterion.
pub const EPSILON: f64 = 1e-12;

/// Errors that can occur while factoring a matrix or solving `Ax = b`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinearAlgebraError {
    /// Matrix or vector sizes are incompatible with the requested operation.
    InvalidDimensions { expected: String, actual: String },
    /// The coefficient matrix is not square.
    NonSquareMatrix { rows: usize, cols: usize },
    /// The coefficient matrix does not have full rank; there is no unique solution.
    SingularMatrix,
    /// A pivot was numerically zero at the given elimination step.
    ZeroPivot { index: usize },
    /// The system `Ax = b` has no solution.
    InconsistentSystem,
    /// The matrix is not symmetric, so Cholesky factorization does not apply.
    NotSymmetric,
    /// The matrix is not positive definite, so Cholesky factorization does not apply.
    NotPositiveDefinite,
    /// A non-finite value was encountered during elimination or substitution.
    NumericalFailure,
}

impl fmt::Display for LinearAlgebraError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimensions { expected, actual } => {
                write!(f, "invalid dimensions: expected {expected}, got {actual}")
            }
            Self::NonSquareMatrix { rows, cols } => {
                write!(f, "matrix is not square: {rows}×{cols}")
            }
            Self::SingularMatrix => write!(f, "matrix is singular"),
            Self::ZeroPivot { index } => {
                write!(f, "zero pivot encountered at index {index}")
            }
            Self::InconsistentSystem => write!(f, "linear system is inconsistent"),
            Self::NotSymmetric => write!(f, "matrix is not symmetric"),
            Self::NotPositiveDefinite => write!(f, "matrix is not positive definite"),
            Self::NumericalFailure => {
                write!(f, "numerical failure: non-finite value encountered")
            }
        }
    }
}

impl std::error::Error for LinearAlgebraError {}

pub(crate) fn is_effectively_zero(x: f64) -> bool {
    x.abs() < EPSILON
}

/// Ensures `A` is a non-empty square matrix.
pub(crate) fn validate_square_matrix(a: &Matrix<f64>) -> Result<usize, LinearAlgebraError> {
    if a.is_empty() {
        return Err(LinearAlgebraError::InvalidDimensions {
            expected: "non-empty square matrix".to_string(),
            actual: format!("{}×{} matrix", a.rows(), a.cols()),
        });
    }

    if a.rows() != a.cols() {
        return Err(LinearAlgebraError::NonSquareMatrix {
            rows: a.rows(),
            cols: a.cols(),
        });
    }

    Ok(a.rows())
}

/// Ensures `A` is square and `b` has one entry per row of `A`.
pub(crate) fn validate_linear_system(
    a: &Matrix<f64>,
    b: &Vector<f64>,
) -> Result<usize, LinearAlgebraError> {
    let n = validate_square_matrix(a)?;

    if b.len() != n {
        return Err(LinearAlgebraError::InvalidDimensions {
            expected: format!("right-hand side of length {n}"),
            actual: format!("length {}", b.len()),
        });
    }

    Ok(n)
}

/// Builds the augmented matrix `[A | b]`.
pub(crate) fn augment(a: &Matrix<f64>, b: &Vector<f64>) -> Vec<Vec<f64>> {
    let rhs = b.as_slice();
    let mut aug = a.to_vec();

    for (row, &bi) in aug.iter_mut().zip(rhs.iter()) {
        row.push(bi);
    }

    aug
}

/// Returns the row `i >= from_row` whose entry in `col` has largest magnitude.
pub(crate) fn find_pivot_row(mat: &[Vec<f64>], from_row: usize, col: usize) -> usize {
    let mut max_row = from_row;
    let mut max_val = mat[from_row][col].abs();

    for (i, row) in mat.iter().enumerate().skip(from_row + 1) {
        let val = row[col].abs();
        if val > max_val {
            max_val = val;
            max_row = i;
        }
    }

    max_row
}

/// Classifies a zero-pivot column of an augmented system as inconsistent or singular.
pub(crate) fn zero_pivot_error(aug: &[Vec<f64>], from_row: usize, n: usize) -> LinearAlgebraError {
    let rhs = n;
    for row in aug.iter().skip(from_row) {
        if !is_effectively_zero(row[rhs]) {
            return LinearAlgebraError::InconsistentSystem;
        }
    }
    LinearAlgebraError::SingularMatrix
}

/// Solves `Ux = c` for upper-triangular `U` by back substitution.
pub(crate) fn back_substitution(
    u: &[Vec<f64>],
    c: &[f64],
) -> Result<Vector<f64>, LinearAlgebraError> {
    let n = c.len();
    let mut x = vec![0.0; n];

    for i in (0..n).rev() {
        let mut sum = c[i];
        for (j, xj) in x.iter().enumerate().skip(i + 1) {
            sum -= u[i][j] * xj;
        }

        let diag = u[i][i];
        if !diag.is_finite() || !sum.is_finite() {
            return Err(LinearAlgebraError::NumericalFailure);
        }
        if is_effectively_zero(diag) {
            if !is_effectively_zero(sum) {
                return Err(LinearAlgebraError::InconsistentSystem);
            }
            return Err(LinearAlgebraError::SingularMatrix);
        }

        x[i] = sum / diag;
    }

    Ok(Vector::new(x))
}

/// Solves `Ly = b` for lower-triangular `L` by forward substitution.
pub(crate) fn forward_substitution(
    l: &[Vec<f64>],
    b: &[f64],
) -> Result<Vec<f64>, LinearAlgebraError> {
    let n = b.len();
    let mut y = vec![0.0; n];

    for i in 0..n {
        let mut sum = b[i];
        for (j, yj) in y.iter().enumerate().take(i) {
            sum -= l[i][j] * yj;
        }

        let diag = l[i][i];
        if !diag.is_finite() || !sum.is_finite() {
            return Err(LinearAlgebraError::NumericalFailure);
        }
        if is_effectively_zero(diag) {
            return Err(LinearAlgebraError::SingularMatrix);
        }

        y[i] = sum / diag;
    }

    Ok(y)
}
