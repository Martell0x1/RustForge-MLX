//! Errors and numerical tolerances shared by linear-algebra algorithms.

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
