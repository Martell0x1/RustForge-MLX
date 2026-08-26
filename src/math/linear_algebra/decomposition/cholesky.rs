//! Cholesky factorization `A = L Lᵀ` for symmetric positive-definite matrices.

use crate::math::linear_algebra::common::{is_effectively_zero, validate_square_matrix};
use crate::math::linear_algebra::{EPSILON, LinearAlgebraError};
use crate::math::matrix::Matrix;

/// Lower-triangular Cholesky factor of a symmetric positive-definite matrix.
///
/// The factor satisfies
///
/// ```text
/// A = L Lᵀ
/// ```
///
/// where `L` is lower-triangular with positive diagonal entries.
#[derive(Debug, Clone)]
pub struct CholeskyDecomposition {
    l: Matrix<f64>,
}

impl CholeskyDecomposition {
    /// Lower-triangular factor `L` such that `A = L Lᵀ`.
    pub fn l(&self) -> &Matrix<f64> {
        &self.l
    }
}

/// Computes the Cholesky factorization `A = L Lᵀ`.
///
/// The algorithm is the Banachiewicz (row-wise) form: for each row `i`,
///
/// ```text
/// L[i, j] = (A[i, j] − Σ_{k<j} L[i, k] L[j, k]) / L[j, j]     (j < i)
/// L[i, i] = √(A[i, i] − Σ_{k<i} L[i, k]²)
/// ```
///
/// `A` must be square, symmetric, and positive definite. There is no pivoting;
/// a non-positive remainder on the diagonal means `A` is not SPD.
///
/// # Errors
///
/// * [`LinearAlgebraError::NonSquareMatrix`] if `A` is not square.
/// * [`LinearAlgebraError::InvalidDimensions`] if `A` is empty.
/// * [`LinearAlgebraError::NotSymmetric`] if `A` is not symmetric within
///   [`EPSILON`].
/// * [`LinearAlgebraError::NotPositiveDefinite`] if a diagonal remainder is
///   non-positive.
/// * [`LinearAlgebraError::NumericalFailure`] if a non-finite value appears.
///
/// # Numerical notes
///
/// Diagonal remainders are tested against zero with [`EPSILON`].
/// Ill-conditioned SPD matrices can still produce a large residual, and a
/// tiny negative remainder from rounding is reported as not positive definite.
pub fn cholesky_decompose(a: &Matrix<f64>) -> Result<CholeskyDecomposition, LinearAlgebraError> {
    let n = validate_square_matrix(a)?;
    let work = a.to_vec();
    ensure_symmetric(&work, n)?;

    let mut l = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..=i {
            let mut sum = work[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }

            if !sum.is_finite() {
                return Err(LinearAlgebraError::NumericalFailure);
            }

            if i == j {
                if sum <= 0.0 || is_effectively_zero(sum) {
                    return Err(LinearAlgebraError::NotPositiveDefinite);
                }
                l[i][i] = sum.sqrt();
                if !l[i][i].is_finite() {
                    return Err(LinearAlgebraError::NumericalFailure);
                }
            } else {
                let diag = l[j][j];
                if is_effectively_zero(diag) {
                    return Err(LinearAlgebraError::NotPositiveDefinite);
                }
                l[i][j] = sum / diag;
                if !l[i][j].is_finite() {
                    return Err(LinearAlgebraError::NumericalFailure);
                }
            }
        }
    }

    Ok(CholeskyDecomposition { l: Matrix::new(l) })
}

fn ensure_symmetric(a: &[Vec<f64>], n: usize) -> Result<(), LinearAlgebraError> {
    for i in 0..n {
        for j in 0..i {
            if !a[i][j].is_finite() || !a[j][i].is_finite() {
                return Err(LinearAlgebraError::NumericalFailure);
            }
            if (a[i][j] - a[j][i]).abs() >= EPSILON {
                return Err(LinearAlgebraError::NotSymmetric);
            }
        }
    }
    Ok(())
}
