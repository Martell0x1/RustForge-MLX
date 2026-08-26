use rustforge_mlx::math::linear_algebra::LinearAlgebraError;
use rustforge_mlx::math::linear_algebra::decomposition::{SVDDecomposition, svd_decompose};
use rustforge_mlx::math::matrix::Matrix;

const TOL: f64 = 1e-8;

fn assert_close(a: f64, b: f64) {
    assert!((a - b).abs() < TOL, "expected {a} ≈ {b} (tol {TOL})");
}

fn assert_matrices_close(a: &Matrix<f64>, b: &Matrix<f64>) {
    assert_eq!(
        a.shape(),
        b.shape(),
        "matrix shapes differ: {:?} vs {:?}",
        a.shape(),
        b.shape()
    );
    for i in 0..a.rows() {
        for j in 0..a.cols() {
            assert_close(*a.get(i, j).unwrap(), *b.get(i, j).unwrap());
        }
    }
}

fn reconstruct(svd: &SVDDecomposition) -> Matrix<f64> {
    svd.u().clone() * svd.sigma().clone() * svd.v().transpose()
}

fn assert_a_equals_usvt(a: &Matrix<f64>, svd: &SVDDecomposition) {
    assert_matrices_close(a, &reconstruct(svd));
}

fn assert_diagonal_nonnegative(sigma: &Matrix<f64>) {
    assert_eq!(
        sigma.rows(),
        sigma.cols(),
        "Σ should be square in the thin SVD"
    );
    for i in 0..sigma.rows() {
        for j in 0..sigma.cols() {
            let value = *sigma.get(i, j).unwrap();
            if i == j {
                assert!(value >= -TOL, "singular value σ[{i}] is negative: {value}");
            } else {
                assert_close(value, 0.0);
            }
        }
    }
}

fn assert_orthogonal(m: &Matrix<f64>) {
    assert_eq!(
        m.rows(),
        m.cols(),
        "orthogonality check requires a square matrix"
    );
    let mtm = m.transpose() * m.clone();
    assert_matrices_close(&mtm, &Matrix::<f64>::identity(m.rows()));
}

fn sorted_singular_values(sigma: &Matrix<f64>) -> Vec<f64> {
    let mut values = sigma.diagonal();
    values.sort_by(|a, b| b.partial_cmp(a).unwrap());
    values
}

#[test]
fn svd_reconstructs_simple_2x2() {
    let a = Matrix::new(vec![vec![3.0, 0.0], vec![0.0, 1.0]]);
    let svd = svd_decompose(&a).unwrap();

    assert_eq!(svd.u().shape(), (2, 2));
    assert_eq!(svd.sigma().shape(), (2, 2));
    assert_eq!(svd.v().shape(), (2, 2));
    assert_diagonal_nonnegative(svd.sigma());
    assert_a_equals_usvt(&a, &svd);
}

#[test]
fn svd_singular_values_match_diagonal_matrix() {
    let a = Matrix::new(vec![vec![3.0, 0.0], vec![0.0, 1.0]]);
    let svd = svd_decompose(&a).unwrap();
    let sigma = sorted_singular_values(svd.sigma());

    assert_close(sigma[0], 3.0);
    assert_close(sigma[1], 1.0);
}

#[test]
fn svd_factors_are_orthogonal_for_full_rank_square() {
    let a = Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    let svd = svd_decompose(&a).unwrap();

    assert_orthogonal(svd.u());
    assert_orthogonal(svd.v());
    assert_diagonal_nonnegative(svd.sigma());
    assert_a_equals_usvt(&a, &svd);
}

#[test]
fn svd_reconstructs_3x3() {
    let a = Matrix::new(vec![
        vec![2.0, 1.0, -1.0],
        vec![-3.0, -1.0, 2.0],
        vec![-2.0, 1.0, 2.0],
    ]);
    let svd = svd_decompose(&a).unwrap();

    assert_eq!(svd.u().shape(), (3, 3));
    assert_eq!(svd.sigma().shape(), (3, 3));
    assert_eq!(svd.v().shape(), (3, 3));
    assert_orthogonal(svd.u());
    assert_orthogonal(svd.v());
    assert_a_equals_usvt(&a, &svd);
}

#[test]
fn svd_of_identity_has_unit_singular_values() {
    let a = Matrix::<f64>::identity(3);
    let svd = svd_decompose(&a).unwrap();

    for value in svd.sigma().diagonal() {
        assert_close(value, 1.0);
    }
    assert_orthogonal(svd.u());
    assert_orthogonal(svd.v());
    assert_a_equals_usvt(&a, &svd);
}

#[test]
fn svd_of_spd_matrix_matches_eigenvalues() {
    // SPD ⇒ singular values equal the eigenvalues of A.
    let a = Matrix::new(vec![vec![4.0, 2.0], vec![2.0, 3.0]]);
    let svd = svd_decompose(&a).unwrap();
    let sigma = sorted_singular_values(svd.sigma());

    assert_close(sigma[0], 5.5615528128);
    assert_close(sigma[1], 1.4384471872);
    assert_a_equals_usvt(&a, &svd);
}

#[test]
fn svd_reconstructs_tall_matrix() {
    let a = Matrix::new(vec![vec![1.0, 1.0], vec![1.0, 0.0], vec![0.0, 1.0]]);
    let svd = svd_decompose(&a).unwrap();

    assert_eq!(svd.u().shape(), (3, 2));
    assert_eq!(svd.sigma().shape(), (2, 2));
    assert_eq!(svd.v().shape(), (2, 2));
    assert_orthogonal(svd.v());
    assert_diagonal_nonnegative(svd.sigma());
    assert_a_equals_usvt(&a, &svd);
}

#[test]
fn svd_reconstructs_wide_matrix() {
    let a = Matrix::new(vec![vec![1.0, 1.0, 0.0], vec![1.0, 0.0, 1.0]]);
    let svd = svd_decompose(&a).unwrap();

    assert_eq!(svd.u().shape(), (2, 3));
    assert_eq!(svd.sigma().shape(), (3, 3));
    assert_eq!(svd.v().shape(), (3, 3));
    assert_orthogonal(svd.v());
    assert_a_equals_usvt(&a, &svd);
}

#[test]
fn svd_reconstructs_rank_deficient_matrix() {
    let a = Matrix::new(vec![vec![1.0, 2.0], vec![2.0, 4.0]]);
    let svd = svd_decompose(&a).unwrap();
    let sigma = sorted_singular_values(svd.sigma());

    assert_close(sigma[1], 0.0);
    assert!(sigma[0] > TOL, "expected a positive leading singular value");
    assert_a_equals_usvt(&a, &svd);
}

#[test]
fn svd_rejects_empty_matrix() {
    let a: Matrix<f64> = Matrix::new(Vec::<Vec<f64>>::new());
    match svd_decompose(&a) {
        Err(LinearAlgebraError::InvalidDimensions { .. }) => {}
        other => panic!("expected InvalidDimensions, got {other:?}"),
    }
}
