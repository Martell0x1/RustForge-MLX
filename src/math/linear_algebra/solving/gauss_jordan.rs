//! Gauss–Jordan elimination with partial pivoting.

use super::{LinearAlgebraError, augment, zero_pivot_error};
use crate::math::linear_algebra::common::{
    find_pivot_row, is_effectively_zero, validate_linear_system,
};
use crate::math::matrix::Matrix;
use crate::math::vector::Vector;

/// Solves the linear system `Ax = b` using Gauss–Jordan elimination with partial pivoting.
///
/// The algorithm forms the augmented matrix `[A | b]` and reduces it to
/// reduced row echelon form `[I | x]`:
///
/// ```text
/// [A | b]
///    ↓
/// [I | x]
/// ```
///
/// The solution is then the last column of the reduced matrix.
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
/// Scaling each pivot row to 1 can amplify rounding error on ill-conditioned systems.
pub fn gauss_jordan_solve(
    a: &Matrix<f64>,
    b: &Vector<f64>,
) -> Result<Vector<f64>, LinearAlgebraError> {
    let n = validate_linear_system(a, b)?;
    let mut aug = augment(a, b);
    reduce_to_rref(&mut aug, n)?;

    Ok(Vector::new(aug.into_iter().map(|row| row[n])))
}

/// Transforms `[A | b]` into `[I | x]` using partial pivoting.
fn reduce_to_rref(aug: &mut [Vec<f64>], n: usize) -> Result<(), LinearAlgebraError> {
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

        for entry in aug[k].iter_mut().skip(k) {
            *entry /= pivot;
        }

        for i in 0..n {
            if i == k {
                continue;
            }

            let factor = aug[i][k];
            if is_effectively_zero(factor) {
                continue;
            }
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
