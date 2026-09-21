use crate::math::vector::Vector;
use std::ops::{Add, Div, Index, IndexMut, Mul, Sub};

#[derive(Debug, Clone, PartialEq)]
/// A two-dimensional collection of values stored as rows.
///
/// `Matrix::new` checks that all rows have the same length. `Matrix::from_slice`
/// copies rows without checking their lengths; pass rectangular data to it because
/// matrix operations assume that every row has the same number of columns.
///
/// Construct a rectangular matrix with [`Matrix::new`]:
///
/// ```
/// use rustforge_mlx::math::matrix::Matrix;
///
/// let matrix = Matrix::new([[1, 2], [3, 4]]);
/// assert_eq!(matrix.shape(), (2, 2));
/// ```
pub struct Matrix<T> {
    data: Vec<Vec<T>>,
}

impl<T> Matrix<T> {
    /// Creates a matrix from rows.
    ///
    /// Rows are stored in the order yielded by `rows`.
    ///
    /// # Panics
    ///
    /// Panics when the rows have different lengths.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustforge_mlx::math::matrix::Matrix;
    ///
    /// let matrix = Matrix::new([[1, 2, 3], [4, 5, 6]]);
    /// assert_eq!(matrix.shape(), (2, 3));
    /// ```
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

    /// Returns the number of rows.
    pub fn rows(&self) -> usize {
        self.data.len()
    }

    /// Returns the number of columns in the first row, or `0` when the matrix has no rows.
    pub fn cols(&self) -> usize {
        self.data.first().map_or(0, |row| row.len())
    }

    /// Returns the matrix dimensions as `(row count, first-row column count)`.
    pub fn shape(&self) -> (usize, usize) {
        (self.rows(), self.cols())
    }

    /// Returns `rows() * cols()`.
    ///
    /// For a rectangular matrix, this is the number of elements.
    pub fn len(&self) -> usize {
        self.rows() * self.cols()
    }

    /// Returns `true` when the matrix has no rows or its first row has no columns.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty() || self.cols() == 0
    }

    /// Returns a reference to the element at `(row, col)`, or `None` if it is out of bounds.
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        self.data.get(row)?.get(col)
    }

    /// Replaces the element at `(row, col)` when that position exists.
    ///
    /// If either index is out of bounds, this method leaves the matrix unchanged.
    pub fn set(&mut self, row: usize, col: usize, value: T) {
        if let Some(r) = self.data.get_mut(row) {
            if let Some(element) = r.get_mut(col) {
                *element = value;
            }
        }
    }

    /// Borrows the matrix rows as a slice.
    pub fn as_slice(&self) -> &[Vec<T>] {
        &self.data
    }

    /// Clones the matrix rows into a new nested vector.
    pub fn to_vec(&self) -> Vec<Vec<T>>
    where
        T: Clone,
    {
        self.data.clone()
    }

    /// Consumes the matrix and returns its rows as a nested vector.
    pub fn into_vec(self) -> Vec<Vec<T>> {
        self.data
    }

    /// Clones rows from borrowed slices into a matrix.
    ///
    /// The supplied rows should have equal lengths, as required by rectangular matrix
    /// operations.
    pub fn from_slice(data: &[&[T]]) -> Self
    where
        T: Clone,
    {
        Self {
            data: data.iter().map(|row| row.to_vec()).collect(),
        }
    }

    /// Converts a vector into a single-column matrix.
    ///
    /// A vector of length `n` becomes a matrix with shape `(n, 1)`.
    pub fn from_vector(vector: Vector<T>) -> Self
    where
        T: Clone,
    {
        let vec_data = Vector::into_vec(vector);
        Self {
            data: vec_data.into_iter().map(|x| vec![x]).collect(),
        }
    }

    /// Returns a clone of the row at `index`, or `None` if it is out of bounds.
    pub fn row(&self, index: usize) -> Option<Vec<T>>
    where
        T: Clone,
    {
        self.data.get(index).cloned()
    }

    /// Returns a clone of the column at `index`, or `None` if it is out of bounds.
    pub fn column(&self, index: usize) -> Option<Vec<T>>
    where
        T: Clone,
    {
        if index >= self.cols() {
            return None;
        }

        Some(self.data.iter().map(|row| row[index].clone()).collect())
    }
    /// Returns a clone of the main diagonal, from the top-left corner.
    ///
    /// The result has `min(rows, columns)` elements for rectangular matrices.
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
    /// Creates a `rows` by `cols` matrix filled with [`Default::default`].
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
    /// Creates a `rows` by `cols` matrix filled with one.
    pub fn ones(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![vec![T::from(1u8); cols]; rows],
        }
    }

    /// Creates a square identity matrix of the given `size`.
    ///
    /// The diagonal contains ones and all other elements contain zero.
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

    /// Adds corresponding elements of two matrices.
    ///
    /// # Panics
    ///
    /// Panics when the matrices have different shapes.
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

    /// Subtracts corresponding elements of `other` from this matrix.
    ///
    /// # Panics
    ///
    /// Panics when the matrices have different shapes.
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

    /// Multiplies two matrices using the standard row-by-column product.
    ///
    /// For shapes `(m, n)` and `(n, p)`, the result has shape `(m, p)`.
    ///
    /// # Panics
    ///
    /// Panics when the left matrix's column count differs from the right matrix's row
    /// count.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustforge_mlx::math::matrix::Matrix;
    ///
    /// let product = Matrix::new([[1, 2], [3, 4]]) * Matrix::new([[5], [6]]);
    /// assert_eq!(product.to_vec(), vec![vec![17], vec![39]]);
    /// ```
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

    /// Divides corresponding elements of two matrices.
    ///
    /// # Panics
    ///
    /// Panics when the matrices have different shapes.
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

    /// Multiplies every element by `scalar`.
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

    /// Divides every element by `scalar`.
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
    /// Returns the transpose, swapping rows and columns.
    ///
    /// An `(m, n)` matrix becomes an `(n, m)` matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustforge_mlx::math::matrix::Matrix;
    ///
    /// let transposed = Matrix::new([[1, 2, 3], [4, 5, 6]]).transpose();
    /// assert_eq!(transposed.to_vec(), vec![vec![1, 4], vec![2, 5], vec![3, 6]]);
    /// ```
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

    /// Returns all elements in row-major order as a vector.
    pub fn flatten(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.data
            .iter()
            .flat_map(|row| row.iter().cloned())
            .collect()
    }

    /// Applies `f` to each element and returns a matrix containing the results.
    ///
    /// The function receives a shared reference to each element.
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

    /// Returns the trace: the sum of the main diagonal of a square matrix.
    ///
    /// # Panics
    ///
    /// Panics when the matrix is not square.
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
    /// Returns the sum of all elements.
    ///
    /// An empty matrix returns [`Default::default`].
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

    /// Returns the arithmetic mean of all elements as `f64`.
    ///
    /// # Panics
    ///
    /// Panics when the matrix is empty.
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

    /// Returns a reference to the smallest element, or `None` for an empty matrix.
    ///
    /// # Panics
    ///
    /// Panics if two values cannot be compared, such as a floating-point `NaN`.
    pub fn min(&self) -> Option<&T>
    where
        T: PartialOrd,
    {
        self.data
            .iter()
            .flat_map(|row| row.iter())
            .min_by(|a, b| a.partial_cmp(b).unwrap())
    }

    /// Returns a reference to the largest element, or `None` for an empty matrix.
    ///
    /// # Panics
    ///
    /// Panics if two values cannot be compared, such as a floating-point `NaN`.
    pub fn max(&self) -> Option<&T>
    where
        T: PartialOrd,
    {
        self.data
            .iter()
            .flat_map(|row| row.iter())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
    }

    /// Returns the `(row, column)` coordinates of the smallest element.
    ///
    /// Returns `None` for an empty matrix.
    ///
    /// # Panics
    ///
    /// Panics if two values cannot be compared, such as a floating-point `NaN`.
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

    /// Returns the `(row, column)` coordinates of the largest element.
    ///
    /// Returns `None` for an empty matrix.
    ///
    /// # Panics
    ///
    /// Panics if two values cannot be compared, such as a floating-point `NaN`.
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
    /// Returns the sum of a row, or `None` if `row` is out of bounds.
    pub fn row_sum(&self, row: usize) -> Option<T>
    where
        T: Add<Output = T> + Default + Copy,
    {
        self.data
            .get(row)
            .map(|values| values.iter().copied().fold(T::default(), |acc, x| acc + x))
    }

    /// Returns the sum of a column, or `None` if `col` is out of bounds.
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

    /// Returns the arithmetic mean of a row as `f64`.
    ///
    /// Returns `None` if `row` is out of bounds or the row is empty.
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

    /// Returns the arithmetic mean of a column as `f64`.
    ///
    /// Returns `None` if `col` is out of bounds or the matrix has no rows.
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
    /// Returns the Frobenius norm: the square root of the sum of squared elements.
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

    /// Returns this matrix converted to `f64` and divided by its Frobenius norm.
    ///
    /// The returned matrix has Frobenius norm one.
    ///
    /// # Panics
    ///
    /// Panics when the matrix has a zero Frobenius norm.
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

impl<T> Index<usize> for Matrix<T> {
    type Output = [T];

    /// Returns the row at `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    fn index(&self, row: usize) -> &Self::Output {
        &self.data[row]
    }
}

impl<T> IndexMut<usize> for Matrix<T> {
    /// Returns a mutable reference to the row at `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    fn index_mut(&mut self, row: usize) -> &mut Self::Output {
        &mut self.data[row]
    }
}
