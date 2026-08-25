//! Householder QR factorization.

use super::{LinearAlgebraError, back_substitution, is_effectively_zero};
use crate::math::matrix::Matrix;
use crate::math::vector::Vector;

/// Factors of a matrix produced by Householder QR.
///
/// The factors satisfy
///
/// ```text
/// A = Q R
/// ```
///
/// where `Q` is orthogonal (`m × m`) and `R` is upper triangular / upper
/// trapezoidal (`m × n`).
#[derive(Debug, Clone)]
pub struct QRDecomposition {
    q: Matrix<f64>,
    r: Matrix<f64>,
}

impl QRDecomposition {
    /// Orthogonal factor `Q`.
    pub fn q(&self) -> &Matrix<f64> {
        &self.q
    }

    /// Upper-triangular (or upper-trapezoidal) factor `R`.
    pub fn r(&self) -> &Matrix<f64> {
        &self.r
    }

    /// Solves `Ax = b` using this factorization: `Rx = Qᵀb`.
    pub fn solve(&self, b: &Vector<f64>) -> Result<Vector<f64>, LinearAlgebraError> {
        qr_solve(self, b)
    }
}

/// Computes the Householder QR factorization of a matrix.
///
/// For each column `k`, a reflector
///
/// ```text
/// H_k = I - 2 u uᵀ
/// ```
///
/// is chosen so that the subcolumn `A[k.., k]` is mapped onto a multiple of
/// `e_1`. The product of these reflectors is `Q`, and the transformed matrix
/// is `R`:
///
/// ```text
/// A = Q R
/// Q = H_0 H_1 ⋯ H_{p-1}
/// R = H_{p-1} ⋯ H_1 H_0 A
/// ```
///
/// Rank-deficient matrices are still factorized; a zero on the diagonal of
/// `R` is reported later by [`qr_solve`].
///
/// # Errors
///
/// * [`LinearAlgebraError::InvalidDimensions`] if `A` is empty.
/// * [`LinearAlgebraError::NumericalFailure`] if a non-finite value appears.
///
/// # Numerical notes
///
/// The Householder vector uses the sign convention `u = x + sign(x_0) ‖x‖ e_1`
/// to avoid cancellation. Pivots / column norms smaller than [`super::EPSILON`]
/// are treated as zero. Householder QR is backward stable, but ill-conditioned
/// systems can still produce a large residual.
pub fn qr_decompose(a: &Matrix<f64>) -> Result<QRDecomposition, LinearAlgebraError> {
    if a.is_empty() {
        return Err(LinearAlgebraError::InvalidDimensions {
            expected: "non-empty matrix".to_string(),
            actual: format!("{}×{} matrix", a.rows(), a.cols()),
        });
    }

    let m = a.rows();
    let n = a.cols();
    let mut q = identity_data(m);
    let mut r = a.to_vec();
    let n_reflectors = m.min(n);

    for k in 0..n_reflectors {
        let x = column_from(&r, k, k);
        let Some((u, alpha)) = householder_unit_vector(&x)? else {
            continue;
        };

        apply_householder_left(&mut r, k, &u, k + 1);
        r[k][k] = alpha;
        for row in r.iter_mut().skip(k + 1) {
            row[k] = 0.0;
        }

        apply_householder_right(&mut q, k, &u);
    }

    Ok(QRDecomposition {
        q: Matrix::new(q),
        r: Matrix::new(r),
    })
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
    let q = decomposition.q.to_vec();
    let r = decomposition.r.to_vec();
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

fn identity_data(n: usize) -> Vec<Vec<f64>> {
    let mut data = vec![vec![0.0; n]; n];
    for (i, row) in data.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    data
}

fn column_from(mat: &[Vec<f64>], from_row: usize, col: usize) -> Vec<f64> {
    mat.iter().skip(from_row).map(|row| row[col]).collect()
}

fn hypot_norm(x: &[f64]) -> f64 {
    x.iter().fold(0.0, |acc, &v| acc.hypot(v))
}

/// Unit Householder vector `u` such that `(I - 2uuᵀ)x = α e₁`,
/// with `α = -sign(x₀) ‖x‖`.
///
/// Returns `None` when `x` is numerically zero.
fn householder_unit_vector(x: &[f64]) -> Result<Option<(Vec<f64>, f64)>, LinearAlgebraError> {
    if x.iter().any(|v| !v.is_finite()) {
        return Err(LinearAlgebraError::NumericalFailure);
    }

    let norm = hypot_norm(x);
    if !norm.is_finite() {
        return Err(LinearAlgebraError::NumericalFailure);
    }
    if is_effectively_zero(norm) {
        return Ok(None);
    }

    let sign = if x[0] >= 0.0 { 1.0 } else { -1.0 };
    let alpha = -sign * norm;
    let mut u = x.to_vec();
    u[0] += sign * norm;

    let u_norm = hypot_norm(&u);
    if !u_norm.is_finite() {
        return Err(LinearAlgebraError::NumericalFailure);
    }
    if is_effectively_zero(u_norm) {
        return Ok(None);
    }

    for ui in &mut u {
        *ui /= u_norm;
    }

    Ok(Some((u, alpha)))
}

/// Left-multiply rows `k..` of `mat` by `H = I - 2uuᵀ`, starting at `col_start`.
fn apply_householder_left(mat: &mut [Vec<f64>], k: usize, u: &[f64], col_start: usize) {
    let n = mat.first().map_or(0, |row| row.len());

    for j in col_start..n {
        let mut dot = 0.0;
        for (i, &ui) in u.iter().enumerate() {
            dot += ui * mat[k + i][j];
        }
        let scale = 2.0 * dot;
        for (i, &ui) in u.iter().enumerate() {
            mat[k + i][j] -= scale * ui;
        }
    }
}

/// Right-multiply `Q` by the embedded reflector `I_k ⊕ (I - 2uuᵀ)`.
fn apply_householder_right(q: &mut [Vec<f64>], k: usize, u: &[f64]) {
    for row in q.iter_mut() {
        let mut dot = 0.0;
        for (t, &ut) in u.iter().enumerate() {
            dot += row[k + t] * ut;
        }
        let scale = 2.0 * dot;
        for (t, &ut) in u.iter().enumerate() {
            row[k + t] -= scale * ut;
        }
    }
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
