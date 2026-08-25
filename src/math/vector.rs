use std::ops::{Add, Div, Mul, Sub};
#[derive(Debug, Clone, PartialEq)]
pub struct Vector<T> {
    data: Vec<T>,
}

impl<T> Add for Vector<T>
where
    T: Add<Output = T>,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        assert_eq!(self.len(), other.len());
        Self::new(self.data.into_iter().zip(other.data).map(|(a, b)| a + b))
    }
}

impl<T> Sub for Vector<T>
where
    T: Sub<Output = T>,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        assert_eq!(self.len(), other.len());

        Self::new(self.data.into_iter().zip(other.data).map(|(a, b)| a - b))
    }
}

impl<T> Mul for Vector<T>
where
    T: Mul<Output = T>,
{
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        assert_eq!(self.len(), other.len());

        Self::new(self.data.into_iter().zip(other.data).map(|(a, b)| a * b))
    }
}
impl<T> Div for Vector<T>
where
    T: Div<Output = T>,
{
    type Output = Self;

    fn div(self, other: Self) -> Self {
        assert_eq!(self.len(), other.len());

        Self::new(self.data.into_iter().zip(other.data).map(|(a, b)| a / b))
    }
}

impl<T> Mul<T> for Vector<T>
where
    T: Mul<Output = T> + Copy,
{
    type Output = Self;

    fn mul(self, scalar: T) -> Self {
        Self::new(self.data.into_iter().map(|x| x * scalar))
    }
}

impl<T> Vector<T> {
    pub fn new<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        Self {
            data: iter.into_iter().collect(),
        }
    }

    pub fn zeros(n: usize) -> Self
    where
        T: Default + Clone,
    {
        Self {
            data: vec![T::default(); n],
        }
    }

    pub fn get(&self, idx: usize) -> Option<T>
    where
        T: Default + Copy,
    {
        self.data.get(idx).copied()
    }

    pub fn set(&mut self, idx: usize, val: T)
    where
        T: Default + Clone,
    {
        if idx >= 0 && idx < self.len() {
            self.data[idx] = val;
        }
    }

    pub fn ones(n: usize) -> Self
    where
        T: From<u8> + Clone,
    {
        Self {
            data: vec![T::from(1u8); n],
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn from_slice(slice: &[T]) -> Self
    where
        T: Clone,
    {
        Self {
            data: slice.to_vec(),
        }
    }

    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    pub fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.data.clone()
    }

    pub fn into_vec(self) -> Vec<T> {
        self.data
    }

    pub fn dot(&self, other: &Self) -> T
    where
        T: Mul<Output = T> + Add<Output = T> + Default + Copy,
    {
        assert_eq!(self.len(), other.len(), "Vector dimensions must match");

        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a * b)
            .fold(T::default(), |acc, x| acc + x)
    }

    pub fn norm(&self) -> f64
    where
        T: Into<f64> + Copy,
    {
        self.data
            .iter()
            .map(|&x| {
                let x: f64 = x.into();
                x * x
            })
            .sum::<f64>()
            .sqrt()
    }

    pub fn normalize(&self) -> Vector<f64>
    where
        T: Into<f64> + Copy,
    {
        let norm = self.norm();

        assert!(norm != 0.0, "Cannot normalize a zero vector");

        Vector::new(self.data.iter().map(|&x| {
            let x: f64 = x.into();
            x / norm
        }))
    }

    pub fn distance(&self, other: &Self) -> f64
    where
        T: Into<f64> + Copy,
    {
        assert_eq!(self.len(), other.len(), "Vector dimensions must match");

        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| {
                let a: f64 = a.into();
                let b: f64 = b.into();

                let diff = a - b;
                diff * diff
            })
            .sum::<f64>()
            .sqrt()
    }

    pub fn sum(&self) -> T
    where
        T: Add<Output = T> + Default + Copy,
    {
        self.data
            .iter()
            .copied()
            .fold(T::default(), |acc, x| acc + x)
    }

    pub fn mean(&self) -> f64
    where
        T: Into<f64> + Copy,
    {
        assert!(!self.is_empty(), "Cannot calculate mean of empty vector");

        self.data.iter().map(|&x| x.into()).sum::<f64>() / self.len() as f64
    }

    pub fn min(&self) -> Option<&T>
    where
        T: PartialOrd,
    {
        self.data.iter().min_by(|a, b| a.partial_cmp(b).unwrap())
    }

    pub fn max(&self) -> Option<&T>
    where
        T: PartialOrd,
    {
        self.data.iter().max_by(|a, b| a.partial_cmp(b).unwrap())
    }

    pub fn argmin(&self) -> Option<usize>
    where
        T: PartialOrd,
    {
        self.data
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
    }

    pub fn argmax(&self) -> Option<usize>
    where
        T: PartialOrd,
    {
        self.data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
    }

    pub fn map<U, F>(&self, f: F) -> Vector<U>
    where
        F: FnMut(&T) -> U,
    {
        Vector::new(self.data.iter().map(f))
    }

    pub fn zip<U>(&self, other: &Vector<U>) -> Vector<(T, U)>
    where
        T: Clone,
        U: Clone,
    {
        assert_eq!(self.len(), other.len(), "Vector dimensions must match");

        Vector::new(self.data.iter().cloned().zip(other.data.iter().cloned()))
    }
}
