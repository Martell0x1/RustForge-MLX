//! Solve `Ax = b` using a Householder QR factorization.

use crate::math::linear_algebra::LinearAlgebraError;
use crate::math::linear_algebra::common::back_substitution;
use crate::math::linear_algebra::decomposition::{QRDecomposition, qr_decompose};
use crate::math::matrix::Matrix;
use crate::math::vector::Vector;

/// Solves `Ax = b` by computing a QR factorization of `A`, then substituting.
///
/// ```text
/// A = Q R
/// R x = Qᵀ b
/// ```
pub fn solve_qr(a: &Matrix<f64>, b: &Vector<f64>) -> Result<Vector<f64>, LinearAlgebraError> {
    qr_solve(&qr_decompose(a)?, b)
}

/// Solves `Ax = b` given a previously computed Householder QR factorization of `A`.
///
/// The square system is reduced to triangular form by orthogonality of `Q`:
///
/// ```text
/// A x = b
/// Q R x = b
/// R x = Qᵀ b
/// ```
///
/// then `x` is recovered by back substitution.
///
/// # Errors
///
/// * [`LinearAlgebraError::NonSquareMatrix`] if the stored factors do not
///   correspond to a square system.
/// * [`LinearAlgebraError::InvalidDimensions`] if `b` does not match the
///   number of rows of `Q`.
/// * [`LinearAlgebraError::SingularMatrix`] if a diagonal entry of `R` is
///   numerically zero.
/// * [`LinearAlgebraError::NumericalFailure`] if a non-finite value appears.
pub fn qr_solve(
    decomposition: &QRDecomposition,
    b: &Vector<f64>,
) -> Result<Vector<f64>, LinearAlgebraError> {
    let q = decomposition.q().to_vec();
    let r = decomposition.r().to_vec();
    let m = q.len();
    let n = r.first().map_or(0, |row| row.len());

    if m != n {
        return Err(LinearAlgebraError::NonSquareMatrix { rows: m, cols: n });
    }
    if b.len() != m {
        return Err(LinearAlgebraError::InvalidDimensions {
            expected: format!("right-hand side of length {m}"),
            actual: format!("length {}", b.len()),
        });
    }

    let y = q_transpose_mul(&q, b.as_slice())?;
    back_substitution(&r, &y)
}

fn q_transpose_mul(q: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>, LinearAlgebraError> {
    let m = q.len();
    let mut y = vec![0.0; m];

    for (j, yj) in y.iter_mut().enumerate() {
        let mut sum = 0.0;
        for (i, row) in q.iter().enumerate() {
            sum += row[j] * b[i];
        }
        if !sum.is_finite() {
            return Err(LinearAlgebraError::NumericalFailure);
        }
        *yj = sum;
    }

    Ok(y)
}
