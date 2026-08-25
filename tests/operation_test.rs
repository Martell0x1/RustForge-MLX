use rustforge_mlx::math::linear_algebra::operations;
use rustforge_mlx::math::matrix::Matrix;
use rustforge_mlx::math::vector::Vector;

#[test]
fn test_vector_matrix_mul() {
    // Arrange
    let matrix = Matrix::new(vec![vec![6, 2, 4], vec![-1, 4, 3], vec![-2, 9, 3]]);

    let vector = Vector::new(vec![4, -2, 1]);

    let result = Matrix::new(vec![vec![24], vec![-9], vec![-23]]);

    // Act
    let operation = operations::vector_matrix_mul(matrix, vector);

    // Assert

    assert_eq!(operation, result);
}
