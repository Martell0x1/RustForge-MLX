//! Solve `Ax = b` using a Cholesky factorization.

use crate::math::linear_algebra::LinearAlgebraError;
use crate::math::linear_algebra::common::{back_substitution, forward_substitution};
use crate::math::linear_algebra::decomposition::{CholeskyDecomposition, cholesky_decompose};
use crate::math::matrix::Matrix;
use crate::math::vector::Vector;

/// Solves `Ax = b` by computing a Cholesky factorization of `A`, then substituting.
///
/// ```text
/// A = L Lᵀ
/// L y = b
/// Lᵀ x = y
/// ```
pub fn solve_cholesky(a: &Matrix<f64>, b: &Vector<f64>) -> Result<Vector<f64>, LinearAlgebraError> {
    cholesky_solve(&cholesky_decompose(a)?, b)
}

/// Solves `Ax = b` given a previously computed Cholesky factorization of `A`.
///
/// The solve uses the two triangular substitutions
///
/// ```text
/// A = L Lᵀ
/// L y = b
/// Lᵀ x = y
/// ```
///
/// # Errors
///
/// * [`LinearAlgebraError::InvalidDimensions`] if `b` does not match the
///   factorization size.
/// * [`LinearAlgebraError::SingularMatrix`] if a diagonal entry of `L` is
///   numerically zero.
/// * [`LinearAlgebraError::NumericalFailure`] if a non-finite value appears.
pub fn cholesky_solve(
    decomposition: &CholeskyDecomposition,
    b: &Vector<f64>,
) -> Result<Vector<f64>, LinearAlgebraError> {
    let n = decomposition.l().rows();
    if b.len() != n {
        return Err(LinearAlgebraError::InvalidDimensions {
            expected: format!("right-hand side of length {n}"),
            actual: format!("length {}", b.len()),
        });
    }

    let l = decomposition.l().to_vec();
    let y = forward_substitution(&l, b.as_slice())?;
    let lt = decomposition.l().transpose().to_vec();
    back_substitution(&lt, &y)
}
