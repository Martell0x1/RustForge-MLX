use crate::math::linear_algebra::LinearAlgebraError;
use crate::math::matrix::Matrix;
use libm::{atan2, cos, sin};

#[derive(Debug, Clone)]
pub struct JacobiEigenDecomposition {
    pub val: Vec<f64>,
    vecs: Matrix<f64>,
}

impl JacobiEigenDecomposition {
    pub fn val(&self) -> &Vec<f64> {
        &self.val
    }

    pub fn vecs(&self) -> &Matrix<f64> {
        &self.vecs
    }
}
pub fn perform_jacobi_eigen_decomposition(
    a: &Matrix<f64>,
    max_iterations: usize,
) -> Result<JacobiEigenDecomposition, LinearAlgebraError> {
    // Validate matrix

    if a.rows() != a.cols() {
        return Err(LinearAlgebraError::InvalidDimensions {
            expected: "square matrix".to_string(),
            actual: format!("{}x{}", a.rows(), a.cols()),
        });
    }

    let n = a.rows();

    if n == 0 {
        return Err(LinearAlgebraError::InvalidDimensions {
            expected: "non-empty matrix".to_string(),
            actual: "0x0".to_string(),
        });
    }

    // Check symmetry

    const TOLERANCE: f64 = 1e-12;

    for i in 0..n {
        for j in 0..i {
            if (a[i][j] - a[j][i]).abs() > TOLERANCE {
                return Err(LinearAlgebraError::NotSymmetric);
            }
        }
    }

    // Working matrix

    let mut a = a.clone();

    // Eigenvector matrix

    let mut v = Matrix::<f64>::identity(n);

    // Jacobi iterations

    for _ in 0..max_iterations {
        // Find largest off-diagonal element

        let (p, q, max_value) = largest_off_diagonal(&a);

        // Converged

        if max_value < TOLERANCE {
            break;
        }

        // Calculate rotation angle

        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];

        let theta = 0.5 * atan2(2.0 * apq, app - aqq);

        let c = cos(theta);
        let s = sin(theta);

        // Apply rotation

        let (a_rotated, j) = apply_rotation(&a, p, q, c, s);

        a = a_rotated;

        // Accumulate eigenvectors
        v = v * j;
    }

    let eigenvalues = a.diagonal();

    let eigenvectors = v.clone();

    Ok(JacobiEigenDecomposition {
        val: eigenvalues,
        vecs: eigenvectors,
    })
}

fn largest_off_diagonal(a: &Matrix<f64>) -> (usize, usize, f64) {
    let n = a.rows();
    let mut p = 0;
    let mut q = 0;
    let mut max_value: f64 = 0.0;

    for i in 0..n {
        for j in (i + 1)..n {
            let value = a[i][j].abs();

            if value > max_value {
                max_value = value;
                p = i;
                q = j;
            }
        }
    }
    (p, q, max_value)
}

fn apply_rotation(
    a: &Matrix<f64>,
    p: usize,
    q: usize,
    c: f64,
    s: f64,
) -> (Matrix<f64>, Matrix<f64>) {
    let n = a.rows();

    let mut j = Matrix::<f64>::identity(n);

    j[p][p] = c;
    j[p][q] = -s;
    j[q][p] = s;
    j[q][q] = c;

    // A' = Jᵀ A J

    let a_rotated = j.transpose() * a.clone() * j.clone();

    (a_rotated, j)
}
