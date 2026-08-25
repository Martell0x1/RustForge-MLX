//! Doolittle LU factorization with partial pivoting.

use super::{
    LinearAlgebraError, back_substitution, find_pivot_row, forward_substitution,
    is_effectively_zero, validate_square_matrix,
};
use crate::math::matrix::Matrix;
use crate::math::vector::Vector;

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

    /// Solves `Ax = b` using this factorization: `Ly = Pb`, then `Ux = y`.
    pub fn solve(&self, b: &Vector<f64>) -> Result<Vector<f64>, LinearAlgebraError> {
        lu_solve(self, b)
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
/// Pivots with absolute value smaller than [`super::EPSILON`] are treated as zero.
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
    let n = decomposition.pivots.len();
    if b.len() != n {
        return Err(LinearAlgebraError::InvalidDimensions {
            expected: format!("right-hand side of length {n}"),
            actual: format!("length {}", b.len()),
        });
    }

    let rhs = b.as_slice();
    let pb: Vec<f64> = decomposition
        .pivots
        .iter()
        .map(|&src_row| rhs[src_row])
        .collect();

    let l = decomposition.l.to_vec();
    let u = decomposition.u.to_vec();
    let y = forward_substitution(&l, &pb)?;
    back_substitution(&u, &y)
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
