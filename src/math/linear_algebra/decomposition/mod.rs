//! Matrix factorizations.
//!
//! This module produces explicit factors. Solving `Ax = b` from those factors
//! lives in [`crate::math::linear_algebra::solving`].
//!
//! ```text
//! PA = LU     Doolittle LU with partial pivoting
//! A  = QR     Householder QR
//! A  = LLᵀ    Cholesky
//! A  = UΣVᵀ   SVD
//! ```

mod cholesky;
pub mod eigen;
mod lu;
mod qr;
mod svd;

pub use cholesky::{CholeskyDecomposition, cholesky_decompose};
pub use lu::{LUDecomposition, lu_decompose};
pub use qr::{QRDecomposition, qr_decompose};
pub use svd::{SVDDecomposition, svd_decompose};
