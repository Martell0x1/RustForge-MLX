use rustforge_mlx::math::linear_algebra::operations;
use rustforge_mlx::math::linear_algebra::solving::{
    CholeskyDecomposition, LUDecomposition, LinearAlgebraError, QRDecomposition,
    cholesky_decompose, cholesky_solve, gauss_jordan_solve, gaussian_solve, lu_decompose, lu_solve,
    qr_decompose, qr_solve,
};
use rustforge_mlx::math::matrix::Matrix;
use rustforge_mlx::math::vector::Vector;

const TOL: f64 = 1e-9;

fn assert_close(a: f64, b: f64) {
    assert!((a - b).abs() < TOL, "expected {a} ≈ {b} (tol {TOL})");
}

fn assert_vectors_close(a: &Vector<f64>, b: &Vector<f64>) {
    assert_eq!(a.len(), b.len(), "vector lengths differ");
    for i in 0..a.len() {
        assert_close(a.get(i).unwrap(), b.get(i).unwrap());
    }
}

fn assert_matrices_close(a: &Matrix<f64>, b: &Matrix<f64>) {
    assert_eq!(a.shape(), b.shape(), "matrix shapes differ");
    for i in 0..a.rows() {
        for j in 0..a.cols() {
            assert_close(*a.get(i, j).unwrap(), *b.get(i, j).unwrap());
        }
    }
}

/// Verifies the mathematical residual `A x ≈ b`.
fn assert_residual(a: &Matrix<f64>, x: &Vector<f64>, b: &Vector<f64>) {
    let ax = operations::vector_matrix_mul(a.clone(), x.clone());
    for i in 0..b.len() {
        assert_close(*ax.get(i, 0).unwrap(), b.get(i).unwrap());
    }
}

fn simple_2x2() -> (Matrix<f64>, Vector<f64>) {
    // 2x + y = 5
    //  x - y = 1
    let a = Matrix::new(vec![vec![2.0, 1.0], vec![1.0, -1.0]]);
    let b = Vector::new(vec![5.0, 1.0]);
    (a, b)
}

fn system_3x3() -> (Matrix<f64>, Vector<f64>) {
    // 2x + y - z = 8
    // -3x - y + 2z = -11
    // -2x + y + 2z = -3
    let a = Matrix::new(vec![
        vec![2.0, 1.0, -1.0],
        vec![-3.0, -1.0, 2.0],
        vec![-2.0, 1.0, 2.0],
    ]);
    let b = Vector::new(vec![8.0, -11.0, -3.0]);
    (a, b)
}

fn needs_row_swap() -> (Matrix<f64>, Vector<f64>) {
    // First pivot is zero, so a row swap is required.
    // x + z = 2
    // y + z = 1
    // x + y = 3
    let a = Matrix::new(vec![
        vec![0.0, 1.0, 1.0],
        vec![1.0, 0.0, 1.0],
        vec![1.0, 1.0, 0.0],
    ]);
    let b = Vector::new(vec![1.0, 2.0, 3.0]);
    (a, b)
}

fn singular_consistent() -> (Matrix<f64>, Vector<f64>) {
    let a = Matrix::new(vec![vec![1.0, 2.0], vec![2.0, 4.0]]);
    let b = Vector::new(vec![3.0, 6.0]);
    (a, b)
}

fn singular_inconsistent() -> (Matrix<f64>, Vector<f64>) {
    let a = Matrix::new(vec![vec![1.0, 2.0], vec![2.0, 4.0]]);
    let b = Vector::new(vec![3.0, 5.0]);
    (a, b)
}

fn assert_no_unique_solution(result: Result<Vector<f64>, LinearAlgebraError>) {
    match result {
        Err(LinearAlgebraError::SingularMatrix)
        | Err(LinearAlgebraError::ZeroPivot { .. })
        | Err(LinearAlgebraError::InconsistentSystem) => {}
        other => panic!("expected a singular/inconsistent error, got {other:?}"),
    }
}

/* -------------------------------------------------------------------------- */
/* Gaussian elimination                                                       */
/* -------------------------------------------------------------------------- */

#[test]
fn gaussian_solves_simple_2x2() {
    let (a, b) = simple_2x2();
    let x = gaussian_solve(&a, &b).unwrap();

    assert_vectors_close(&x, &Vector::new(vec![2.0, 1.0]));
    assert_residual(&a, &x, &b);
}

#[test]
fn gaussian_solves_3x3() {
    let (a, b) = system_3x3();
    let x = gaussian_solve(&a, &b).unwrap();

    assert_vectors_close(&x, &Vector::new(vec![2.0, 3.0, -1.0]));
    assert_residual(&a, &x, &b);
}

#[test]
fn gaussian_swaps_rows_when_pivot_is_zero() {
    let (a, b) = needs_row_swap();
    let x = gaussian_solve(&a, &b).unwrap();

    assert_vectors_close(&x, &Vector::new(vec![2.0, 1.0, 0.0]));
    assert_residual(&a, &x, &b);
}

#[test]
fn gaussian_rejects_singular_matrix() {
    let (a, b) = singular_consistent();
    assert_no_unique_solution(gaussian_solve(&a, &b));
}

#[test]
fn gaussian_rejects_inconsistent_system() {
    let (a, b) = singular_inconsistent();
    let err = gaussian_solve(&a, &b).unwrap_err();
    assert_eq!(err, LinearAlgebraError::InconsistentSystem);
}

#[test]
fn gaussian_rejects_non_square_matrix() {
    let a = Matrix::new(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    let b = Vector::new(vec![1.0, 2.0]);
    let err = gaussian_solve(&a, &b).unwrap_err();
    assert_eq!(
        err,
        LinearAlgebraError::NonSquareMatrix { rows: 2, cols: 3 }
    );
}

#[test]
fn gaussian_rejects_incompatible_rhs() {
    let a = Matrix::new(vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
    let b = Vector::new(vec![1.0, 2.0, 3.0]);
    match gaussian_solve(&a, &b) {
        Err(LinearAlgebraError::InvalidDimensions { .. }) => {}
        other => panic!("expected InvalidDimensions, got {other:?}"),
    }
}

/* -------------------------------------------------------------------------- */
/* Gauss–Jordan elimination                                                   */
/* -------------------------------------------------------------------------- */

#[test]
fn gauss_jordan_solves_simple_2x2() {
    let (a, b) = simple_2x2();
    let x = gauss_jordan_solve(&a, &b).unwrap();

    assert_vectors_close(&x, &Vector::new(vec![2.0, 1.0]));
    assert_residual(&a, &x, &b);
}

#[test]
fn gauss_jordan_solves_3x3() {
    let (a, b) = system_3x3();
    let x = gauss_jordan_solve(&a, &b).unwrap();

    assert_vectors_close(&x, &Vector::new(vec![2.0, 3.0, -1.0]));
    assert_residual(&a, &x, &b);
}

#[test]
fn gauss_jordan_swaps_rows_when_pivot_is_zero() {
    let (a, b) = needs_row_swap();
    let x = gauss_jordan_solve(&a, &b).unwrap();

    assert_vectors_close(&x, &Vector::new(vec![2.0, 1.0, 0.0]));
    assert_residual(&a, &x, &b);
}

#[test]
fn gauss_jordan_rejects_singular_matrix() {
    let (a, b) = singular_consistent();
    assert_no_unique_solution(gauss_jordan_solve(&a, &b));
}

#[test]
fn gauss_jordan_rejects_inconsistent_system() {
    let (a, b) = singular_inconsistent();
    let err = gauss_jordan_solve(&a, &b).unwrap_err();
    assert_eq!(err, LinearAlgebraError::InconsistentSystem);
}

#[test]
fn gauss_jordan_rejects_non_square_matrix() {
    let a = Matrix::new(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    let b = Vector::new(vec![1.0, 2.0]);
    let err = gauss_jordan_solve(&a, &b).unwrap_err();
    assert_eq!(
        err,
        LinearAlgebraError::NonSquareMatrix { rows: 2, cols: 3 }
    );
}

#[test]
fn gauss_jordan_rejects_incompatible_rhs() {
    let a = Matrix::new(vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
    let b = Vector::new(vec![1.0]);
    match gauss_jordan_solve(&a, &b) {
        Err(LinearAlgebraError::InvalidDimensions { .. }) => {}
        other => panic!("expected InvalidDimensions, got {other:?}"),
    }
}

/* -------------------------------------------------------------------------- */
/* LU decomposition                                                           */
/* -------------------------------------------------------------------------- */

fn assert_unit_lower_triangular(l: &Matrix<f64>) {
    assert_eq!(l.rows(), l.cols());
    for i in 0..l.rows() {
        for j in 0..l.cols() {
            let value = *l.get(i, j).unwrap();
            if i == j {
                assert_close(value, 1.0);
            } else if i < j {
                assert_close(value, 0.0);
            }
        }
    }
}

fn assert_upper_triangular(u: &Matrix<f64>) {
    for i in 0..u.rows() {
        for j in 0..u.cols() {
            if i > j {
                assert_close(*u.get(i, j).unwrap(), 0.0);
            }
        }
    }
}

fn assert_pa_equals_lu(a: &Matrix<f64>, decomposition: &LUDecomposition) {
    let pa = decomposition.p().clone() * a.clone();
    let lu = decomposition.l().clone() * decomposition.u().clone();
    assert_matrices_close(&pa, &lu);
}

#[test]
fn lu_factors_have_correct_structure() {
    let (a, _) = system_3x3();
    let decomposition = lu_decompose(&a).unwrap();

    assert_unit_lower_triangular(decomposition.l());
    assert_upper_triangular(decomposition.u());
}

#[test]
fn lu_satisfies_pa_equals_lu() {
    let (a, _) = system_3x3();
    let decomposition = lu_decompose(&a).unwrap();
    assert_pa_equals_lu(&a, &decomposition);
}

#[test]
fn lu_permutation_is_identity_when_no_swap_is_needed() {
    let a = Matrix::new(vec![vec![2.0, 1.0], vec![1.0, -1.0]]);
    let decomposition = lu_decompose(&a).unwrap();
    assert_matrices_close(decomposition.p(), &Matrix::<f64>::identity(2));
    assert_pa_equals_lu(&a, &decomposition);
}

#[test]
fn lu_applies_permutation_when_pivoting() {
    let (a, _) = needs_row_swap();
    let decomposition = lu_decompose(&a).unwrap();

    let identity = Matrix::<f64>::identity(3);
    let p = decomposition.p();
    assert_eq!(p.shape(), (3, 3));
    assert_ne!(p.to_vec(), identity.to_vec());
    assert_pa_equals_lu(&a, &decomposition);
}

#[test]
fn lu_solves_ax_equals_b() {
    let (a, b) = simple_2x2();
    let decomposition = lu_decompose(&a).unwrap();
    let x = lu_solve(&decomposition, &b).unwrap();

    assert_vectors_close(&x, &Vector::new(vec![2.0, 1.0]));
    assert_residual(&a, &x, &b);
}

#[test]
fn lu_solves_3x3_and_pivoting_system() {
    let (a, b) = system_3x3();
    let x = lu_decompose(&a).unwrap().solve(&b).unwrap();
    assert_residual(&a, &x, &b);

    let (a, b) = needs_row_swap();
    let x = lu_decompose(&a).unwrap().solve(&b).unwrap();
    assert_vectors_close(&x, &Vector::new(vec![2.0, 1.0, 0.0]));
    assert_residual(&a, &x, &b);
}

#[test]
fn lu_rejects_singular_matrix() {
    let (a, _) = singular_consistent();
    match lu_decompose(&a) {
        Err(LinearAlgebraError::SingularMatrix) | Err(LinearAlgebraError::ZeroPivot { .. }) => {}
        other => panic!("expected singular factorization, got {other:?}"),
    }
}

#[test]
fn lu_solve_rejects_incompatible_rhs() {
    let (a, _) = simple_2x2();
    let decomposition = lu_decompose(&a).unwrap();
    let b = Vector::new(vec![1.0]);
    match lu_solve(&decomposition, &b) {
        Err(LinearAlgebraError::InvalidDimensions { .. }) => {}
        other => panic!("expected InvalidDimensions, got {other:?}"),
    }
}

#[test]
fn all_solvers_agree_on_the_same_system() {
    let (a, b) = system_3x3();

    let x_ge = gaussian_solve(&a, &b).unwrap();
    let x_gj = gauss_jordan_solve(&a, &b).unwrap();
    let x_lu = lu_decompose(&a).unwrap().solve(&b).unwrap();
    let x_qr = qr_decompose(&a).unwrap().solve(&b).unwrap();

    assert_vectors_close(&x_ge, &x_gj);
    assert_vectors_close(&x_ge, &x_lu);
    assert_vectors_close(&x_ge, &x_qr);
    assert_residual(&a, &x_ge, &b);
}

/* -------------------------------------------------------------------------- */
/* Householder QR                                                             */
/* -------------------------------------------------------------------------- */

fn assert_orthogonal(q: &Matrix<f64>) {
    assert_eq!(q.rows(), q.cols(), "Q must be square");
    let qtq = q.transpose() * q.clone();
    assert_matrices_close(&qtq, &Matrix::<f64>::identity(q.rows()));
}

fn assert_a_equals_qr(a: &Matrix<f64>, decomposition: &QRDecomposition) {
    let qr = decomposition.q().clone() * decomposition.r().clone();
    assert_matrices_close(a, &qr);
}

#[test]
fn qr_factors_have_correct_structure() {
    let (a, _) = system_3x3();
    let decomposition = qr_decompose(&a).unwrap();

    assert_eq!(decomposition.q().shape(), (3, 3));
    assert_eq!(decomposition.r().shape(), (3, 3));
    assert_orthogonal(decomposition.q());
    assert_upper_triangular(decomposition.r());
}

#[test]
fn qr_satisfies_a_equals_qr() {
    let (a, _) = system_3x3();
    let decomposition = qr_decompose(&a).unwrap();
    assert_a_equals_qr(&a, &decomposition);
}

#[test]
fn qr_handles_leading_zero_without_row_swaps() {
    let (a, b) = needs_row_swap();
    let decomposition = qr_decompose(&a).unwrap();

    assert_orthogonal(decomposition.q());
    assert_upper_triangular(decomposition.r());
    assert_a_equals_qr(&a, &decomposition);

    let x = decomposition.solve(&b).unwrap();
    assert_vectors_close(&x, &Vector::new(vec![2.0, 1.0, 0.0]));
    assert_residual(&a, &x, &b);
}

#[test]
fn qr_factorizes_tall_matrix() {
    let a = Matrix::new(vec![vec![1.0, 1.0], vec![1.0, 0.0], vec![0.0, 1.0]]);
    let decomposition = qr_decompose(&a).unwrap();

    assert_eq!(decomposition.q().shape(), (3, 3));
    assert_eq!(decomposition.r().shape(), (3, 2));
    assert_orthogonal(decomposition.q());
    assert_upper_triangular(decomposition.r());
    assert_a_equals_qr(&a, &decomposition);
}

#[test]
fn qr_solves_simple_2x2() {
    let (a, b) = simple_2x2();
    let decomposition = qr_decompose(&a).unwrap();
    let x = qr_solve(&decomposition, &b).unwrap();

    assert_vectors_close(&x, &Vector::new(vec![2.0, 1.0]));
    assert_residual(&a, &x, &b);
}

#[test]
fn qr_solves_3x3() {
    let (a, b) = system_3x3();
    let x = qr_decompose(&a).unwrap().solve(&b).unwrap();

    assert_vectors_close(&x, &Vector::new(vec![2.0, 3.0, -1.0]));
    assert_residual(&a, &x, &b);
}

#[test]
fn qr_solve_rejects_singular_matrix() {
    let (a, b) = singular_consistent();
    let decomposition = qr_decompose(&a).unwrap();
    assert_no_unique_solution(qr_solve(&decomposition, &b));
}

#[test]
fn qr_solve_rejects_rectangular_factors() {
    let a = Matrix::new(vec![vec![1.0, 1.0], vec![1.0, 0.0], vec![0.0, 1.0]]);
    let decomposition = qr_decompose(&a).unwrap();
    let b = Vector::new(vec![1.0, 2.0, 3.0]);
    let err = qr_solve(&decomposition, &b).unwrap_err();
    assert_eq!(
        err,
        LinearAlgebraError::NonSquareMatrix { rows: 3, cols: 2 }
    );
}

#[test]
fn qr_rejects_empty_matrix() {
    let a: Matrix<f64> = Matrix::new(Vec::<Vec<f64>>::new());
    match qr_decompose(&a) {
        Err(LinearAlgebraError::InvalidDimensions { .. }) => {}
        other => panic!("expected InvalidDimensions, got {other:?}"),
    }
}

#[test]
fn qr_solve_rejects_incompatible_rhs() {
    let (a, _) = simple_2x2();
    let decomposition = qr_decompose(&a).unwrap();
    let b = Vector::new(vec![1.0]);
    match qr_solve(&decomposition, &b) {
        Err(LinearAlgebraError::InvalidDimensions { .. }) => {}
        other => panic!("expected InvalidDimensions, got {other:?}"),
    }
}

/* -------------------------------------------------------------------------- */
/* Cholesky                                                                   */
/* -------------------------------------------------------------------------- */

fn spd_2x2() -> (Matrix<f64>, Vector<f64>) {
    // [[4, 2], [2, 3]] x = [8, 8] → x = [1, 2]
    let a = Matrix::new(vec![vec![4.0, 2.0], vec![2.0, 3.0]]);
    let b = Vector::new(vec![8.0, 8.0]);
    (a, b)
}

fn spd_3x3() -> (Matrix<f64>, Vector<f64>) {
    // Classic SPD example with integer Cholesky factor
    // L = [[2, 0, 0], [6, 1, 0], [-8, 5, 3]]
    let a = Matrix::new(vec![
        vec![4.0, 12.0, -16.0],
        vec![12.0, 37.0, -43.0],
        vec![-16.0, -43.0, 98.0],
    ]);
    let b = Vector::new(vec![0.0, 9.0, 11.0]);
    (a, b)
}

fn assert_lower_triangular_positive_diag(l: &Matrix<f64>) {
    assert_eq!(l.rows(), l.cols());
    for i in 0..l.rows() {
        for j in 0..l.cols() {
            let value = *l.get(i, j).unwrap();
            if i < j {
                assert_close(value, 0.0);
            } else if i == j {
                assert!(
                    value > 0.0,
                    "Cholesky diagonal must be positive, got {value}"
                );
            }
        }
    }
}

fn assert_a_equals_llt(a: &Matrix<f64>, decomposition: &CholeskyDecomposition) {
    let llt = decomposition.l().clone() * decomposition.l().transpose();
    assert_matrices_close(a, &llt);
}

#[test]
fn cholesky_factors_have_correct_structure() {
    let (a, _) = spd_3x3();
    let decomposition = cholesky_decompose(&a).unwrap();

    assert_lower_triangular_positive_diag(decomposition.l());
    assert_vectors_close(
        &Vector::new(decomposition.l().diagonal()),
        &Vector::new(vec![2.0, 1.0, 3.0]),
    );
}

#[test]
fn cholesky_satisfies_a_equals_llt() {
    let (a, _) = spd_3x3();
    let decomposition = cholesky_decompose(&a).unwrap();
    assert_a_equals_llt(&a, &decomposition);
}

#[test]
fn cholesky_factorizes_identity() {
    let a = Matrix::<f64>::identity(3);
    let decomposition = cholesky_decompose(&a).unwrap();
    assert_matrices_close(decomposition.l(), &a);
    assert_a_equals_llt(&a, &decomposition);
}

#[test]
fn cholesky_solves_simple_2x2() {
    let (a, b) = spd_2x2();
    let decomposition = cholesky_decompose(&a).unwrap();
    let x = cholesky_solve(&decomposition, &b).unwrap();

    assert_vectors_close(&x, &Vector::new(vec![1.0, 2.0]));
    assert_residual(&a, &x, &b);
}

#[test]
fn cholesky_solves_3x3() {
    let (a, b) = spd_3x3();
    let x = cholesky_decompose(&a).unwrap().solve(&b).unwrap();
    assert_residual(&a, &x, &b);
}

#[test]
fn cholesky_rejects_non_symmetric_matrix() {
    let a = Matrix::new(vec![vec![2.0, 1.0], vec![0.0, 2.0]]);
    let err = cholesky_decompose(&a).unwrap_err();
    assert_eq!(err, LinearAlgebraError::NotSymmetric);
}

#[test]
fn cholesky_rejects_indefinite_matrix() {
    let a = Matrix::new(vec![vec![1.0, 2.0], vec![2.0, 1.0]]);
    let err = cholesky_decompose(&a).unwrap_err();
    assert_eq!(err, LinearAlgebraError::NotPositiveDefinite);
}

#[test]
fn cholesky_rejects_singular_positive_semidefinite_matrix() {
    let a = Matrix::new(vec![vec![1.0, 1.0], vec![1.0, 1.0]]);
    let err = cholesky_decompose(&a).unwrap_err();
    assert_eq!(err, LinearAlgebraError::NotPositiveDefinite);
}

#[test]
fn cholesky_rejects_non_square_matrix() {
    let a = Matrix::new(vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]]);
    let err = cholesky_decompose(&a).unwrap_err();
    assert_eq!(
        err,
        LinearAlgebraError::NonSquareMatrix { rows: 2, cols: 3 }
    );
}

#[test]
fn cholesky_solve_rejects_incompatible_rhs() {
    let (a, _) = spd_2x2();
    let decomposition = cholesky_decompose(&a).unwrap();
    let b = Vector::new(vec![1.0]);
    match cholesky_solve(&decomposition, &b) {
        Err(LinearAlgebraError::InvalidDimensions { .. }) => {}
        other => panic!("expected InvalidDimensions, got {other:?}"),
    }
}
