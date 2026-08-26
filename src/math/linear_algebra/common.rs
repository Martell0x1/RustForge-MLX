//! Shared numerical helpers for decompositions and solvers.

use super::error::{EPSILON, LinearAlgebraError};
use crate::math::matrix::Matrix;
use crate::math::vector::Vector;

pub(crate) fn is_effectively_zero(x: f64) -> bool {
    x.abs() < EPSILON
}

/// Ensures `A` is a non-empty square matrix.
pub(crate) fn validate_square_matrix(a: &Matrix<f64>) -> Result<usize, LinearAlgebraError> {
    if a.is_empty() {
        return Err(LinearAlgebraError::InvalidDimensions {
            expected: "non-empty square matrix".to_string(),
            actual: format!("{}×{} matrix", a.rows(), a.cols()),
        });
    }

    if a.rows() != a.cols() {
        return Err(LinearAlgebraError::NonSquareMatrix {
            rows: a.rows(),
            cols: a.cols(),
        });
    }

    Ok(a.rows())
}

/// Ensures `A` is square and `b` has one entry per row of `A`.
pub(crate) fn validate_linear_system(
    a: &Matrix<f64>,
    b: &Vector<f64>,
) -> Result<usize, LinearAlgebraError> {
    let n = validate_square_matrix(a)?;

    if b.len() != n {
        return Err(LinearAlgebraError::InvalidDimensions {
            expected: format!("right-hand side of length {n}"),
            actual: format!("length {}", b.len()),
        });
    }

    Ok(n)
}

/// Returns the row `i >= from_row` whose entry in `col` has largest magnitude.
pub(crate) fn find_pivot_row(mat: &[Vec<f64>], from_row: usize, col: usize) -> usize {
    let mut max_row = from_row;
    let mut max_val = mat[from_row][col].abs();

    for (i, row) in mat.iter().enumerate().skip(from_row + 1) {
        let val = row[col].abs();
        if val > max_val {
            max_val = val;
            max_row = i;
        }
    }

    max_row
}

/// Solves `Ux = c` for upper-triangular `U` by back substitution.
pub(crate) fn back_substitution(
    u: &[Vec<f64>],
    c: &[f64],
) -> Result<Vector<f64>, LinearAlgebraError> {
    let n = c.len();
    let mut x = vec![0.0; n];

    for i in (0..n).rev() {
        let mut sum = c[i];
        for (j, xj) in x.iter().enumerate().skip(i + 1) {
            sum -= u[i][j] * xj;
        }

        let diag = u[i][i];
        if !diag.is_finite() || !sum.is_finite() {
            return Err(LinearAlgebraError::NumericalFailure);
        }
        if is_effectively_zero(diag) {
            if !is_effectively_zero(sum) {
                return Err(LinearAlgebraError::InconsistentSystem);
            }
            return Err(LinearAlgebraError::SingularMatrix);
        }

        x[i] = sum / diag;
    }

    Ok(Vector::new(x))
}

/// Solves `Ly = b` for lower-triangular `L` by forward substitution.
pub(crate) fn forward_substitution(
    l: &[Vec<f64>],
    b: &[f64],
) -> Result<Vec<f64>, LinearAlgebraError> {
    let n = b.len();
    let mut y = vec![0.0; n];

    for i in 0..n {
        let mut sum = b[i];
        for (j, yj) in y.iter().enumerate().take(i) {
            sum -= l[i][j] * yj;
        }

        let diag = l[i][i];
        if !diag.is_finite() || !sum.is_finite() {
            return Err(LinearAlgebraError::NumericalFailure);
        }
        if is_effectively_zero(diag) {
            return Err(LinearAlgebraError::SingularMatrix);
        }

        y[i] = sum / diag;
    }

    Ok(y)
}
