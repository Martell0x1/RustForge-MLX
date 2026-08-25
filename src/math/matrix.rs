use crate::math::vector::Vector;
use std::ops::{Add, Div, Mul, Sub};

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix<T> {
    data: Vec<Vec<T>>,
}

impl<T> Matrix<T> {
    /// Create a matrix from rows.
    ///
    /// All rows must have the same number of columns.
    pub fn new<I, R>(rows: I) -> Self
    where
        I: IntoIterator<Item = R>,
        R: IntoIterator<Item = T>,
    {
        let data: Vec<Vec<T>> = rows
            .into_iter()
            .map(|row| row.into_iter().collect())
            .collect();

        if !data.is_empty() {
            let cols = data[0].len();

            assert!(
                data.iter().all(|row| row.len() == cols),
                "All matrix rows must have the same length"
            );
        }

        Self { data }
    }

    pub fn rows(&self) -> usize {
        self.data.len()
    }

    pub fn cols(&self) -> usize {
        self.data.first().map_or(0, |row| row.len())
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.rows(), self.cols())
    }

    pub fn len(&self) -> usize {
        self.rows() * self.cols()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty() || self.cols() == 0
    }

    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        self.data.get(row)?.get(col)
    }

    pub fn set(&mut self, row: usize, col: usize, value: T) {
        if let Some(r) = self.data.get_mut(row) {
            if let Some(element) = r.get_mut(col) {
                *element = value;
            }
        }
    }

    pub fn as_slice(&self) -> &[Vec<T>] {
        &self.data
    }

    pub fn to_vec(&self) -> Vec<Vec<T>>
    where
        T: Clone,
    {
        self.data.clone()
    }

    pub fn into_vec(self) -> Vec<Vec<T>> {
        self.data
    }

    pub fn from_slice(data: &[&[T]]) -> Self
    where
        T: Clone,
    {
        Self {
            data: data.iter().map(|row| row.to_vec()).collect(),
        }
    }

    pub fn from_vector(vector: Vector<T>) -> Self
    where
        T: Clone,
    {
        let vec_data = Vector::into_vec(vector);
        Self {
            data: vec_data.into_iter().map(|x| vec![x]).collect(),
        }
    }

    pub fn row(&self, index: usize) -> Option<Vec<T>>
    where
        T: Clone,
    {
        self.data.get(index).cloned()
    }

    pub fn column(&self, index: usize) -> Option<Vec<T>>
    where
        T: Clone,
    {
        if index >= self.cols() {
            return None;
        }

        Some(self.data.iter().map(|row| row[index].clone()).collect())
    }
    pub fn diagonal(&self) -> Vec<T>
    where
        T: Clone,
    {
        let size = self.rows().min(self.cols());

        (0..size).map(|i| self.data[i][i].clone()).collect()
    }
}

/* -------------------------------------------------------------------------- */
/* Constructors                                                               */
/* -------------------------------------------------------------------------- */

impl<T> Matrix<T>
where
    T: Default + Clone,
{
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![vec![T::default(); cols]; rows],
        }
    }
}

impl<T> Matrix<T>
where
    T: From<u8> + Clone,
{
    pub fn ones(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![vec![T::from(1u8); cols]; rows],
        }
    }

    pub fn identity(size: usize) -> Self {
        let mut data = vec![vec![T::from(0u8); size]; size];

        for i in 0..size {
            data[i][i] = T::from(1u8);
        }

        Self { data }
    }
}

/* -------------------------------------------------------------------------- */
/* Element-wise arithmetic                                                    */
/* -------------------------------------------------------------------------- */

impl<T> Add for Matrix<T>
where
    T: Add<Output = T>,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        assert_eq!(self.shape(), other.shape(), "Matrix dimensions must match");

        Self::new(
            self.data
                .into_iter()
                .zip(other.data)
                .map(|(row_a, row_b)| row_a.into_iter().zip(row_b).map(|(a, b)| a + b)),
        )
    }
}

impl<T> Sub for Matrix<T>
where
    T: Sub<Output = T>,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        assert_eq!(self.shape(), other.shape(), "Matrix dimensions must match");

        Self::new(
            self.data
                .into_iter()
                .zip(other.data)
                .map(|(row_a, row_b)| row_a.into_iter().zip(row_b).map(|(a, b)| a - b)),
        )
    }
}

impl<T> Mul for Matrix<T>
where
    T: Mul<Output = T> + Add<Output = T> + Default + Copy,
{
    type Output = Self;

    /// Matrix multiplication.
    ///
    /// (m x n) * (n x p) = (m x p)
    fn mul(self, other: Self) -> Self {
        assert_eq!(
            self.cols(),
            other.rows(),
            "Matrix dimensions are incompatible for multiplication"
        );

        let rows = self.rows();
        let cols = other.cols();
        let mut result = Self::zeros(rows, cols);

        for i in 0..rows {
            for j in 0..cols {
                let mut sum = T::default();
                for k in 0..self.cols() {
                    sum = sum + self.data[i][k] * other.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        result
    }
}
impl<T> Div for Matrix<T>
where
    T: Div<Output = T>,
{
    type Output = Self;

    /// Element-wise division.
    fn div(self, other: Self) -> Self {
        assert_eq!(self.shape(), other.shape(), "Matrix dimensions must match");

        Self::new(
            self.data
                .into_iter()
                .zip(other.data)
                .map(|(row_a, row_b)| row_a.into_iter().zip(row_b).map(|(a, b)| a / b)),
        )
    }
}

/* -------------------------------------------------------------------------- */
/* Scalar arithmetic                                                          */
/* -------------------------------------------------------------------------- */

impl<T> Mul<T> for Matrix<T>
where
    T: Mul<Output = T> + Copy,
{
    type Output = Self;

    fn mul(self, scalar: T) -> Self {
        Self::new(
            self.data
                .into_iter()
                .map(|row| row.into_iter().map(|x| x * scalar)),
        )
    }
}

impl<T> Div<T> for Matrix<T>
where
    T: Div<Output = T> + Copy,
{
    type Output = Self;

    fn div(self, scalar: T) -> Self {
        Self::new(
            self.data
                .into_iter()
                .map(|row| row.into_iter().map(|x| x / scalar)),
        )
    }
}

/* -------------------------------------------------------------------------- */
/* Matrix operations                                                          */
/* -------------------------------------------------------------------------- */

impl<T> Matrix<T> {
    /// Transpose the matrix.
    pub fn transpose(&self) -> Self
    where
        T: Clone,
    {
        if self.is_empty() {
            return Self::new(Vec::<Vec<T>>::new());
        }
        Self::new(
            (0..self.cols())
                .map(|col| (0..self.rows()).map(move |row| self.data[row][col].clone())),
        )
    }

    /// Flatten matrix into a vector.
    pub fn flatten(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.data
            .iter()
            .flat_map(|row| row.iter().cloned())
            .collect()
    }

    /// Apply a function to every element.
    pub fn map<U, F>(&self, mut f: F) -> Matrix<U>
    where
        F: FnMut(&T) -> U,
    {
        Matrix::new(
            self.data
                .iter()
                .map(|row| row.iter().map(&mut f).collect::<Vec<U>>()),
        )
    }

    /// Return the trace of a square matrix.
    pub fn trace(&self) -> T
    where
        T: Add<Output = T> + Default + Copy,
    {
        assert_eq!(self.rows(), self.cols(), "Trace requires a square matrix");

        self.diagonal()
            .into_iter()
            .fold(T::default(), |acc, x| acc + x)
    }
}

/* -------------------------------------------------------------------------- */
/* Aggregations                                                               */
/* -------------------------------------------------------------------------- */

impl<T> Matrix<T> {
    pub fn sum(&self) -> T
    where
        T: Add<Output = T> + Default + Copy,
    {
        self.data
            .iter()
            .flat_map(|row| row.iter())
            .copied()
            .fold(T::default(), |acc, x| acc + x)
    }

    pub fn mean(&self) -> f64
    where
        T: Into<f64> + Copy,
    {
        assert!(!self.is_empty(), "Cannot calculate mean of empty matrix");

        self.data
            .iter()
            .flat_map(|row| row.iter())
            .map(|&x| x.into())
            .sum::<f64>()
            / self.len() as f64
    }

    pub fn min(&self) -> Option<&T>
    where
        T: PartialOrd,
    {
        self.data
            .iter()
            .flat_map(|row| row.iter())
            .min_by(|a, b| a.partial_cmp(b).unwrap())
    }

    pub fn max(&self) -> Option<&T>
    where
        T: PartialOrd,
    {
        self.data
            .iter()
            .flat_map(|row| row.iter())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
    }

    pub fn argmin(&self) -> Option<(usize, usize)>
    where
        T: PartialOrd,
    {
        self.data
            .iter()
            .enumerate()
            .flat_map(|(row_idx, row)| {
                row.iter()
                    .enumerate()
                    .map(move |(col_idx, value)| (row_idx, col_idx, value))
            })
            .min_by(|(_, _, a), (_, _, b)| a.partial_cmp(b).unwrap())
            .map(|(row, col, _)| (row, col))
    }

    pub fn argmax(&self) -> Option<(usize, usize)>
    where
        T: PartialOrd,
    {
        self.data
            .iter()
            .enumerate()
            .flat_map(|(row_idx, row)| {
                row.iter()
                    .enumerate()
                    .map(move |(col_idx, value)| (row_idx, col_idx, value))
            })
            .max_by(|(_, _, a), (_, _, b)| a.partial_cmp(b).unwrap())
            .map(|(row, col, _)| (row, col))
    }
}

/* -------------------------------------------------------------------------- */
/* Row / Column aggregations                                                  */
/* -------------------------------------------------------------------------- */

impl<T> Matrix<T> {
    pub fn row_sum(&self, row: usize) -> Option<T>
    where
        T: Add<Output = T> + Default + Copy,
    {
        self.data
            .get(row)
            .map(|values| values.iter().copied().fold(T::default(), |acc, x| acc + x))
    }

    pub fn column_sum(&self, col: usize) -> Option<T>
    where
        T: Add<Output = T> + Default + Copy,
    {
        if col >= self.cols() {
            return None;
        }

        Some(
            self.data
                .iter()
                .map(|row| row[col])
                .fold(T::default(), |acc, x| acc + x),
        )
    }

    pub fn row_mean(&self, row: usize) -> Option<f64>
    where
        T: Into<f64> + Copy,
    {
        let values = self.data.get(row)?;

        if values.is_empty() {
            return None;
        }

        Some(values.iter().map(|&x| x.into()).sum::<f64>() / values.len() as f64)
    }

    pub fn column_mean(&self, col: usize) -> Option<f64>
    where
        T: Into<f64> + Copy,
    {
        if col >= self.cols() || self.rows() == 0 {
            return None;
        }

        Some(self.data.iter().map(|row| row[col].into()).sum::<f64>() / self.rows() as f64)
    }
}

/* -------------------------------------------------------------------------- */
/* Norms                                                                       */
/* -------------------------------------------------------------------------- */

impl<T> Matrix<T> {
    /// Frobenius norm.
    pub fn norm(&self) -> f64
    where
        T: Into<f64> + Copy,
    {
        self.data
            .iter()
            .flat_map(|row| row.iter())
            .map(|&x| {
                let x: f64 = x.into();
                x * x
            })
            .sum::<f64>()
            .sqrt()
    }

    /// Normalize the matrix by its Frobenius norm.
    pub fn normalize(&self) -> Matrix<f64>
    where
        T: Into<f64> + Copy,
    {
        let norm = self.norm();

        assert!(norm != 0.0, "Cannot normalize a zero matrix");

        self.map(|x| {
            let x: f64 = (*x).into();
            x / norm
        })
    }
}
