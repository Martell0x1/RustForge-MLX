use rustforge_mlx::math::matrix::Matrix;

#[test]
fn matrix_has_correct_dimensions() {
    let matrix = Matrix::new(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);

    assert_eq!(matrix.rows(), 2);
    assert_eq!(matrix.cols(), 3);
    assert_eq!(matrix.shape(), (2, 3));
}

#[test]
fn matrix_can_be_created_from_rows() {
    let matrix = Matrix::new(vec![vec![1, 2], vec![3, 4], vec![5, 6]]);

    assert_eq!(matrix.to_vec(), vec![vec![1, 2], vec![3, 4], vec![5, 6],]);
}

#[test]
fn matrix_can_be_empty() {
    let matrix: Matrix<i32> = Matrix::new(Vec::<Vec<i32>>::new());

    assert!(matrix.is_empty());
    assert_eq!(matrix.rows(), 0);
    assert_eq!(matrix.cols(), 0);
    assert_eq!(matrix.len(), 0);
}

#[test]
fn matrix_returns_zeros() {
    let matrix = Matrix::<f64>::zeros(2, 3);

    let expected = vec![vec![0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0]];

    assert_eq!(matrix.rows(), 2);
    assert_eq!(matrix.cols(), 3);
    assert_eq!(matrix.to_vec(), expected);
}

#[test]
fn matrix_returns_ones() {
    let matrix = Matrix::<f64>::ones(2, 3);

    let expected = vec![vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0]];

    assert_eq!(matrix.rows(), 2);
    assert_eq!(matrix.cols(), 3);
    assert_eq!(matrix.to_vec(), expected);
}

#[test]
fn matrix_returns_identity() {
    let matrix = Matrix::<i32>::identity(3);

    let expected = vec![vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]];

    assert_eq!(matrix.to_vec(), expected);
}

#[test]
fn matrix_can_get_element() {
    let matrix = Matrix::new(vec![vec![1, 2, 3], vec![4, 5, 6]]);

    assert_eq!(matrix.get(0, 0), Some(&1));
    assert_eq!(matrix.get(0, 2), Some(&3));
    assert_eq!(matrix.get(1, 1), Some(&5));

    assert_eq!(matrix.get(2, 0), None);
    assert_eq!(matrix.get(0, 3), None);
}

#[test]
fn matrix_can_set_element() {
    let mut matrix = Matrix::new(vec![vec![1, 2], vec![3, 4]]);

    matrix.set(1, 1, 10);

    assert_eq!(matrix.get(1, 1), Some(&10));
}

#[test]
fn matrix_can_get_row() {
    let matrix = Matrix::new(vec![vec![1, 2, 3], vec![4, 5, 6]]);

    assert_eq!(matrix.row(0), Some(vec![1, 2, 3]));
    assert_eq!(matrix.row(1), Some(vec![4, 5, 6]));
    assert_eq!(matrix.row(2), None);
}

#[test]
fn matrix_can_get_column() {
    let matrix = Matrix::new(vec![vec![1, 2, 3], vec![4, 5, 6]]);

    assert_eq!(matrix.column(0), Some(vec![1, 4]));
    assert_eq!(matrix.column(1), Some(vec![2, 5]));
    assert_eq!(matrix.column(2), Some(vec![3, 6]));
    assert_eq!(matrix.column(3), None);
}

#[test]
fn matrix_can_get_diagonal() {
    let matrix = Matrix::new(vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]]);

    assert_eq!(matrix.diagonal(), vec![1, 5, 9]);
}

#[test]
fn matrix_addition() {
    let matrix1 = Matrix::new(vec![vec![1, 2], vec![3, 4]]);

    let matrix2 = Matrix::new(vec![vec![5, 6], vec![7, 8]]);

    let result = matrix1 + matrix2;

    let expected = Matrix::new(vec![vec![6, 8], vec![10, 12]]);

    assert_eq!(result, expected);
}

#[test]
fn matrix_subtraction() {
    let matrix1 = Matrix::new(vec![vec![5, 6], vec![7, 8]]);

    let matrix2 = Matrix::new(vec![vec![1, 2], vec![3, 4]]);

    let result = matrix1 - matrix2;

    let expected = Matrix::new(vec![vec![4, 4], vec![4, 4]]);

    assert_eq!(result, expected);
}

#[test]
fn matrix_multiplication() {
    let matrix1 = Matrix::new(vec![vec![1, 2, 3], vec![4, 5, 6]]);

    let matrix2 = Matrix::new(vec![vec![7, 8], vec![9, 10], vec![11, 12]]);

    let result = matrix1 * matrix2;

    let expected = Matrix::new(vec![vec![58, 64], vec![139, 154]]);

    assert_eq!(result, expected);
}

#[test]
fn matrix_multiplication_with_square_matrices() {
    let matrix1 = Matrix::new(vec![vec![1, 2], vec![3, 4]]);

    let matrix2 = Matrix::new(vec![vec![5, 6], vec![7, 8]]);

    let result = matrix1 * matrix2;

    let expected = Matrix::new(vec![vec![19, 22], vec![43, 50]]);

    assert_eq!(result, expected);
}

#[test]
fn matrix_scalar_multiplication() {
    let matrix = Matrix::new(vec![vec![1, 2], vec![3, 4]]);

    let result = matrix * 5;

    let expected = Matrix::new(vec![vec![5, 10], vec![15, 20]]);

    assert_eq!(result, expected);
}

#[test]
fn matrix_division() {
    let matrix1 = Matrix::new(vec![vec![10, 20], vec![30, 40]]);

    let matrix2 = Matrix::new(vec![vec![2, 4], vec![5, 8]]);

    let result = matrix1 / matrix2;

    let expected = Matrix::new(vec![vec![5, 5], vec![6, 5]]);

    assert_eq!(result, expected);
}

#[test]
fn matrix_scalar_division() {
    let matrix = Matrix::new(vec![vec![10, 20], vec![30, 40]]);

    let result = matrix / 10;

    let expected = Matrix::new(vec![vec![1, 2], vec![3, 4]]);

    assert_eq!(result, expected);
}

#[test]
fn matrix_transpose() {
    let matrix = Matrix::new(vec![vec![1, 2, 3], vec![4, 5, 6]]);

    let result = matrix.transpose();

    let expected = Matrix::new(vec![vec![1, 4], vec![2, 5], vec![3, 6]]);

    assert_eq!(result, expected);
}

#[test]
fn matrix_transpose_square_matrix() {
    let matrix = Matrix::new(vec![vec![1, 2], vec![3, 4]]);

    let result = matrix.transpose();

    let expected = Matrix::new(vec![vec![1, 3], vec![2, 4]]);

    assert_eq!(result, expected);
}

#[test]
fn matrix_flatten() {
    let matrix = Matrix::new(vec![vec![1, 2, 3], vec![4, 5, 6]]);

    assert_eq!(matrix.flatten(), vec![1, 2, 3, 4, 5, 6]);
}

#[test]
fn matrix_trace() {
    let matrix = Matrix::new(vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]]);

    assert_eq!(matrix.trace(), 15);
}

#[test]
fn matrix_sum() {
    let matrix = Matrix::new(vec![vec![1, 2], vec![3, 4]]);

    assert_eq!(matrix.sum(), 10);
}

#[test]
fn matrix_mean() {
    let matrix = Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);

    assert_eq!(matrix.mean(), 2.5);
}

#[test]
fn matrix_min() {
    let matrix = Matrix::new(vec![vec![4, 2], vec![8, 1]]);

    assert_eq!(matrix.min(), Some(&1));
}

#[test]
fn matrix_max() {
    let matrix = Matrix::new(vec![vec![4, 2], vec![8, 1]]);

    assert_eq!(matrix.max(), Some(&8));
}

#[test]
fn matrix_argmin() {
    let matrix = Matrix::new(vec![vec![4, 2], vec![8, 1]]);

    assert_eq!(matrix.argmin(), Some((1, 1)));
}

#[test]
fn matrix_argmax() {
    let matrix = Matrix::new(vec![vec![4, 2], vec![8, 1]]);

    assert_eq!(matrix.argmax(), Some((1, 0)));
}

#[test]
fn matrix_map() {
    let matrix = Matrix::new(vec![vec![1, 2], vec![3, 4]]);

    let result = matrix.map(|x| x * x);

    let expected = Matrix::new(vec![vec![1, 4], vec![9, 16]]);

    assert_eq!(result, expected);
}

#[test]
fn matrix_row_sum() {
    let matrix = Matrix::new(vec![vec![1, 2, 3], vec![4, 5, 6]]);

    assert_eq!(matrix.row_sum(0), Some(6));
    assert_eq!(matrix.row_sum(1), Some(15));
    assert_eq!(matrix.row_sum(2), None);
}

#[test]
fn matrix_column_sum() {
    let matrix = Matrix::new(vec![vec![1, 2, 3], vec![4, 5, 6]]);

    assert_eq!(matrix.column_sum(0), Some(5));
    assert_eq!(matrix.column_sum(1), Some(7));
    assert_eq!(matrix.column_sum(2), Some(9));
    assert_eq!(matrix.column_sum(3), None);
}

#[test]
fn matrix_row_mean() {
    let matrix = Matrix::new(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);

    assert_eq!(matrix.row_mean(0), Some(2.0));
    assert_eq!(matrix.row_mean(1), Some(5.0));
    assert_eq!(matrix.row_mean(2), None);
}

#[test]
fn matrix_column_mean() {
    let matrix = Matrix::new(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);

    assert_eq!(matrix.column_mean(0), Some(2.5));
    assert_eq!(matrix.column_mean(1), Some(3.5));
    assert_eq!(matrix.column_mean(2), Some(4.5));
}

#[test]
fn matrix_frobenius_norm() {
    let matrix = Matrix::new(vec![vec![3.0, 0.0], vec![0.0, 4.0]]);

    assert_eq!(matrix.norm(), 5.0);
}

#[test]
fn matrix_normalize() {
    let matrix = Matrix::new(vec![vec![3.0, 0.0], vec![0.0, 4.0]]);

    let result = matrix.normalize();

    let expected = Matrix::new(vec![vec![0.6, 0.0], vec![0.0, 0.8]]);

    assert_eq!(result, expected);
}
