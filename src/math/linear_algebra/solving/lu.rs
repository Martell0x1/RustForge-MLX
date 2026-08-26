//! Solve `Ax = b` using an LU factorization.

use crate::math::linear_algebra::LinearAlgebraError;
use crate::math::linear_algebra::common::{back_substitution, forward_substitution};
use crate::math::linear_algebra::decomposition::{LUDecomposition, lu_decompose};
use crate::math::matrix::Matrix;
use crate::math::vector::Vector;

/// Solves `Ax = b` by computing an LU factorization of `A`, then substituting.
///
/// ```text
/// P A = L U
/// L y = P b
/// U x = y
/// ```
pub fn solve_lu(a: &Matrix<f64>, b: &Vector<f64>) -> Result<Vector<f64>, LinearAlgebraError> {
    lu_solve(&lu_decompose(a)?, b)
}

/// Solves `Ax = b` given a previously computed LU factorization of `A`.
///
/// The solve applies the stored permutation, then performs forward substitution
/// on `L` and back substitution on `U`:
///
/// ```text
/// P A = L U
/// L y = P b
/// U x = y
/// ```
///
/// # Errors
///
/// * [`LinearAlgebraError::InvalidDimensions`] if `b` does not match the
///   factorization size.
/// * [`LinearAlgebraError::SingularMatrix`] if a diagonal entry of `L` or `U`
///   is numerically zero.
/// * [`LinearAlgebraError::NumericalFailure`] if a non-finite value appears.
pub fn lu_solve(
    decomposition: &LUDecomposition,
    b: &Vector<f64>,
) -> Result<Vector<f64>, LinearAlgebraError> {
    let n = decomposition.pivots().len();
    if b.len() != n {
        return Err(LinearAlgebraError::InvalidDimensions {
            expected: format!("right-hand side of length {n}"),
            actual: format!("length {}", b.len()),
        });
    }

    let rhs = b.as_slice();
    let pb: Vec<f64> = decomposition
        .pivots()
        .iter()
        .map(|&src_row| rhs[src_row])
        .collect();

    let l = decomposition.l().to_vec();
    let u = decomposition.u().to_vec();
    let y = forward_substitution(&l, &pb)?;
    back_substitution(&u, &y)
}
