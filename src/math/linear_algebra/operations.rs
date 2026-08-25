use crate::math::matrix::Matrix;
use crate::math::vector::Vector;
use std::ops::{Add, Mul};

pub fn vector_matrix_mul<T>(matrix: Matrix<T>, vector: Vector<T>) -> Matrix<T>
where
    T: Clone + Mul<Output = T> + Add<Output = T> + Default + Copy,
{
    // mxn * nx1 (column vector)
    let vecotr_to_matrix = Matrix::from_vector(vector);

    matrix * vecotr_to_matrix
}
