//! Linear-system solvers for `Ax = b`.
//!
//! Direct elimination methods live here. Factorization-based solvers
//! ([`solve_lu`], [`solve_qr`], [`solve_cholesky`]) compute factors from
//! [`crate::math::linear_algebra::decomposition`] and then apply triangular
//! substitution.

mod cholesky;
mod gauss_jordan;
mod gaussian;
mod lu;
mod qr;

pub use cholesky::{cholesky_solve, solve_cholesky};
pub use gauss_jordan::gauss_jordan_solve;
pub use gaussian::gaussian_solve;
pub use lu::{lu_solve, solve_lu};
pub use qr::{qr_solve, solve_qr};

pub use super::{EPSILON, LinearAlgebraError};

use crate::math::linear_algebra::common::is_effectively_zero;
use crate::math::matrix::Matrix;
use crate::math::vector::Vector;

/// Builds the augmented matrix `[A | b]`.
pub(crate) fn augment(a: &Matrix<f64>, b: &Vector<f64>) -> Vec<Vec<f64>> {
    let rhs = b.as_slice();
    let mut aug = a.to_vec();

    for (row, &bi) in aug.iter_mut().zip(rhs.iter()) {
        row.push(bi);
    }

    aug
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
