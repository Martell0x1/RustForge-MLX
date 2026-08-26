//! Gaussian elimination with partial pivoting.

use super::{LinearAlgebraError, augment, zero_pivot_error};
use crate::math::linear_algebra::common::{
    back_substitution, find_pivot_row, is_effectively_zero, validate_linear_system,
};
use crate::math::matrix::Matrix;
use crate::math::vector::Vector;

/// Solves the linear system `Ax = b` using Gaussian elimination with partial pivoting.
///
/// The algorithm forms the augmented matrix `[A | b]`, reduces it to upper-triangular
/// form `[U | c]` by row swaps and elimination, then recovers `x` by back substitution:
///
/// ```text
/// A | b
///   ↓  elimination
/// U | c
///   ↓  back substitution
///   x
/// ```
///
/// # Errors
///
/// * [`LinearAlgebraError::NonSquareMatrix`] if `A` is not square.
/// * [`LinearAlgebraError::InvalidDimensions`] if `A` is empty or `b` does not
///   have one entry per row of `A`.
/// * [`LinearAlgebraError::SingularMatrix`] if `A` is numerically singular.
/// * [`LinearAlgebraError::InconsistentSystem`] if the system has no solution.
/// * [`LinearAlgebraError::NumericalFailure`] if a non-finite value appears.
///
/// # Numerical notes
///
/// Pivots with absolute value smaller than [`crate::math::linear_algebra::EPSILON`] are treated as zero.
/// Ill-conditioned matrices may still produce inaccurate solutions even when a
/// unique solution exists mathematically.
pub fn gaussian_solve(a: &Matrix<f64>, b: &Vector<f64>) -> Result<Vector<f64>, LinearAlgebraError> {
    let n = validate_linear_system(a, b)?;
    let mut aug = augment(a, b);
    eliminate(&mut aug, n)?;

    let rhs: Vec<f64> = aug.iter().map(|row| row[n]).collect();
    back_substitution(&aug, &rhs)
}

/// Transforms `[A | b]` into `[U | c]` using partial pivoting.
fn eliminate(aug: &mut [Vec<f64>], n: usize) -> Result<(), LinearAlgebraError> {
    for k in 0..n {
        let pivot_row = find_pivot_row(aug, k, k);
        aug.swap(k, pivot_row);

        let pivot = aug[k][k];
        if !pivot.is_finite() {
            return Err(LinearAlgebraError::NumericalFailure);
        }
        if is_effectively_zero(pivot) {
            return Err(zero_pivot_error(aug, k, n));
        }

        for i in (k + 1)..n {
            let factor = aug[i][k] / pivot;
            if !factor.is_finite() {
                return Err(LinearAlgebraError::NumericalFailure);
            }

            for j in k..=n {
                aug[i][j] -= factor * aug[k][j];
            }
        }
    }

    Ok(())
}
