use rustforge_mlx::math::linear_algebra::decomposition::eigen::jacobi;
use rustforge_mlx::math::matrix::Matrix;

#[test]
fn test_jaobi_decomposition_returns_right_eigen_values() {
    // Arrange

    //let mut a = Matrix::<f64>::identity(2);
    let mut a = Matrix::<f64>::new(vec![vec![4.0, 2.0], vec![2.0, 3.0]]);

    let values = jacobi::perform_jacobi_eigen_decomposition(&a, 100).unwrap();

    assert!((values.val[0] - 5.5615528128).abs() < 1e-10);
    assert!((values.val[1] - 1.4384471872).abs() < 1e-10);
}
