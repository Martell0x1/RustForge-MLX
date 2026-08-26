use crate::math::linear_algebra::LinearAlgebraError;
use crate::math::linear_algebra::decomposition::eigen::jacobi::perform_jacobi_eigen_decomposition;
use crate::math::matrix::Matrix;

#[derive(Debug, Clone)]
pub struct SVDDecomposition {
    u: Matrix<f64>,
    sigma: Matrix<f64>,
    v: Matrix<f64>,
}

impl SVDDecomposition {
    pub fn u(&self) -> &Matrix<f64> {
        &self.u
    }
    pub fn sigma(&self) -> &Matrix<f64> {
        &self.sigma
    }
    pub fn v(&self) -> &Matrix<f64> {
        &self.v
    }
}

pub fn svd_decompose(a: &Matrix<f64>) -> Result<SVDDecomposition, LinearAlgebraError> {
    // 1. AᵀA
    let a = a.clone();
    let ata = a.transpose() * a.clone();

    // 2. Eigen decomposition
    let eigens = perform_jacobi_eigen_decomposition(&ata, 100)?;

    // 3. Eigenvalues
    let eigenvalues = eigens.val();

    // 4. Eigenvectors
    let v = eigens.vecs().clone();

    // 5. Singular values
    let singular_values: Vec<f64> = eigenvalues
        .iter()
        .map(|lambda| lambda.max(0.0).sqrt())
        .collect();

    // 6. U = AV / Σ
    let u = compute_u(&a, &v, &singular_values);

    // 7. Build Σ as an n×n diagonal so the thin factorization A = U Σ Vᵀ
    //    is dimensionally consistent: U is m×n, Σ is n×n, V is n×n.
    let n = a.cols();
    let mut sigma = Matrix::<f64>::zeros(n, n);

    for i in 0..n {
        sigma[i][i] = singular_values[i];
    }

    Ok(SVDDecomposition { u, sigma, v })
}

fn compute_u(a: &Matrix<f64>, v: &Matrix<f64>, singular_values: &[f64]) -> Matrix<f64> {
    let av = a.clone() * v.clone();

    let mut u = av.clone();

    for j in 0..u.cols() {
        let sigma = singular_values[j];

        if sigma > 1e-12 {
            for i in 0..u.rows() {
                u[i][j] /= sigma;
            }
        }
    }

    u
}
