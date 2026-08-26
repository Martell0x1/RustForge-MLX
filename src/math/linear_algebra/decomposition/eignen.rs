use crate::math::linear_algebra::LinearAlgebraError;
use crate::math::matrix::Matrix;

#[derive(Debug, Clone)]
pub struct EigenDecomposition {
    eigenvalues: Vec<f64>,
    eigenvectors: Matrix<f64>,
}

impl EigenDecomposition {
    pub fn eigenvalues(&self) -> &Vec<f64> {
        &self.eigenvalues
    }
    pub fn eigenvectors(&self) -> &Matrix<f64> {
        &self.eigenvectors
    }
}

fn ensure_symmetric(a: &Matrix<f64>) -> Result<(), LinearAlgebraError> {
    if a.rows() != a.cols() {
        return Err(LinearAlgebraError::InvalidDimensions {
            expected: "square matrix".to_string(),
            actual: format!("{}x{}", a.rows(), a.cols()),
        });
    }

    for i in 0..a.rows() {
        for j in 0..(i + 1) {
            if a[i][j] != a[j][i] {
                return Err(LinearAlgebraError::NonSquareMatrix {
                    rows: a.rows(),
                    cols: a.cols(),
                });
            }
        }
    }
    Ok(())
}

pub fn eigen_decomposition(a: &Matrix<f64>) -> Result<EigenDecomposition, LinearAlgebraError> {}

