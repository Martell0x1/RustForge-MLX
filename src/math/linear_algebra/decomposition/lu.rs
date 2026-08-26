//! Doolittle LU factorization with partial pivoting.

use crate::math::linear_algebra::LinearAlgebraError;
use crate::math::linear_algebra::common::{
    find_pivot_row, is_effectively_zero, validate_square_matrix,
};
use crate::math::matrix::Matrix;

/// Factors of a square matrix produced by LU decomposition with partial pivoting.
///
/// The factors satisfy
///
/// ```text
/// P A = L U
/// ```
///
/// where `L` is unit lower-triangular (Doolittle form), `U` is upper-triangular,
/// and `P` is a permutation matrix encoding the row swaps.
#[derive(Debug, Clone)]
pub struct LUDecomposition {
    l: Matrix<f64>,
    u: Matrix<f64>,
    p: Matrix<f64>,
    /// `pivots[i]` is the original row index that ended up in position `i`.
    pivots: Vec<usize>,
}

impl LUDecomposition {
    /// Unit lower-triangular factor `L`.
    pub fn l(&self) -> &Matrix<f64> {
        &self.l
    }

    /// Upper-triangular factor `U`.
    pub fn u(&self) -> &Matrix<f64> {
        &self.u
    }

    /// Permutation matrix `P` such that `P A = L U`.
    pub fn p(&self) -> &Matrix<f64> {
        &self.p
    }

    /// Row permutation used to form `P`: `pivots[i]` is the original row that
    /// ended up in position `i`.
    pub(crate) fn pivots(&self) -> &[usize] {
        &self.pivots
    }
}

/// Computes the Doolittle LU factorization of a square matrix with partial pivoting.
///
/// Row swaps are recorded in the permutation matrix `P`, so the identity
/// `P A = L U` holds even when a zero appears on the diagonal of `A`.
///
/// # Errors
///
/// * [`LinearAlgebraError::NonSquareMatrix`] if `A` is not square.
/// * [`LinearAlgebraError::InvalidDimensions`] if `A` is empty.
/// * [`LinearAlgebraError::SingularMatrix`] if `A` is numerically singular.
/// * [`LinearAlgebraError::NumericalFailure`] if a non-finite value appears.
///
/// # Numerical notes
///
/// Pivots with absolute value smaller than [`crate::math::linear_algebra::EPSILON`] are treated as zero.
/// The factorization uses partial pivoting, which is more stable than unpivoted LU
/// but still sensitive to ill-conditioned matrices.
pub fn lu_decompose(a: &Matrix<f64>) -> Result<LUDecomposition, LinearAlgebraError> {
    let n = validate_square_matrix(a)?;
    let mut work = a.to_vec();
    let mut pivots: Vec<usize> = (0..n).collect();

    for k in 0..n {
        let pivot_row = find_pivot_row(&work, k, k);
        work.swap(k, pivot_row);
        pivots.swap(k, pivot_row);

        let pivot = work[k][k];
        if !pivot.is_finite() {
            return Err(LinearAlgebraError::NumericalFailure);
        }
        if is_effectively_zero(pivot) {
            return Err(LinearAlgebraError::SingularMatrix);
        }

        for i in (k + 1)..n {
            work[i][k] /= pivot;
            let factor = work[i][k];
            if !factor.is_finite() {
                return Err(LinearAlgebraError::NumericalFailure);
            }

            for j in (k + 1)..n {
                work[i][j] -= factor * work[k][j];
            }
        }
    }

    Ok(extract_factors(work, pivots, n))
}

fn extract_factors(work: Vec<Vec<f64>>, pivots: Vec<usize>, n: usize) -> LUDecomposition {
    let mut l_data = vec![vec![0.0; n]; n];
    let mut u_data = vec![vec![0.0; n]; n];
    let mut p_data = vec![vec![0.0; n]; n];

    for i in 0..n {
        l_data[i][i] = 1.0;
        p_data[i][pivots[i]] = 1.0;

        for j in 0..n {
            if i > j {
                l_data[i][j] = work[i][j];
            } else {
                u_data[i][j] = work[i][j];
            }
        }
    }

    LUDecomposition {
        l: Matrix::new(l_data),
        u: Matrix::new(u_data),
        p: Matrix::new(p_data),
        pivots,
    }
}
