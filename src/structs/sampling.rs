use std::{
    fmt,
    ops::{Add, Deref, Sub},
};

use nalgebra::SVector;

use crate::structs::Result;

/// A system under test whose output can be classified as "target" or "non-target"
/// behavior. For example, safe/unsafe.
pub trait Classifier<const N: usize> {
    fn classify(&mut self, p: SVector<f64, N>) -> Result<Sample<N>>;
}

/// A Classifier defined by a function (p: SVector) -> Result<bool>
pub struct FunctionClassifier<F, const N: usize>
where
    F: FnMut(SVector<f64, N>) -> Result<bool>,
{
    fut: F,
}

impl<F, const N: usize> FunctionClassifier<F, N>
where
    F: FnMut(SVector<f64, N>) -> Result<bool>,
{
    pub fn new(fut: F) -> Self {
        Self { fut }
    }
}

impl<F, const N: usize> Classifier<N> for FunctionClassifier<F, N>
where
    F: FnMut(SVector<f64, N>) -> Result<bool>,
{
    fn classify(&mut self, p: SVector<f64, N>) -> Result<Sample<N>> {
        Ok(Sample::from_class(p, (self.fut)(p)?))
    }
}

/// A point that falls within the target performance mode, i.e. when classifying this
/// point results in true classification.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WithinMode<const N: usize>(pub SVector<f64, N>);

/// A point that falls outside the target performance mode, i.e. when classifying
/// this point results in false classification.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OutOfMode<const N: usize>(pub SVector<f64, N>);

/// A sample from the system under test's input space with a corresponding target
/// performance classification.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Sample<const N: usize> {
    WithinMode(WithinMode<N>),
    OutOfMode(OutOfMode<N>),
}

impl<const N: usize> Sample<N> {
    pub fn from_class(p: SVector<f64, N>, cls: bool) -> Self {
        if cls {
            Sample::WithinMode(WithinMode(p))
        } else {
            Sample::OutOfMode(OutOfMode(p))
        }
    }

    /// Strips the point of the semantics, returning the raw SVector
    pub fn into_inner(self) -> SVector<f64, N> {
        match self {
            Sample::WithinMode(WithinMode(p)) => p,
            Sample::OutOfMode(OutOfMode(p)) => p,
        }
    }

    pub fn class(&self) -> bool {
        match self {
            Sample::WithinMode(_) => true,
            Sample::OutOfMode(_) => false,
        }
    }
}

impl<const N: usize> fmt::Display for Sample<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self {
            Sample::WithinMode(WithinMode(t)) => write!(f, "WithinMode(p: {:?})", t),
            Sample::OutOfMode(OutOfMode(x)) => write!(f, "OutOfMode(p: {:?})", x),
        }
    }
}

impl<const N: usize> Deref for Sample<N> {
    type Target = SVector<f64, N>;

    fn deref(&self) -> &Self::Target {
        match self {
            Sample::WithinMode(WithinMode(t)) => t,
            Sample::OutOfMode(OutOfMode(x)) => x,
        }
    }
}

impl<const N: usize> Deref for WithinMode<N> {
    type Target = SVector<f64, N>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<const N: usize> Deref for OutOfMode<N> {
    type Target = SVector<f64, N>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// Operators for samples
// # Borrowed
// ## Addition
impl<const N: usize> Add<&WithinMode<N>> for &WithinMode<N> {
    type Output = SVector<f64, N>;

    fn add(self, rhs: &WithinMode<N>) -> Self::Output {
        self.0 + rhs.0
    }
}
impl<const N: usize> Add<&OutOfMode<N>> for &WithinMode<N> {
    type Output = SVector<f64, N>;

    fn add(self, rhs: &OutOfMode<N>) -> Self::Output {
        self.0 + rhs.0
    }
}
impl<const N: usize> Add<&WithinMode<N>> for &SVector<f64, N> {
    type Output = SVector<f64, N>;

    fn add(self, rhs: &WithinMode<N>) -> Self::Output {
        self + rhs.0
    }
}
impl<const N: usize> Add<&OutOfMode<N>> for &SVector<f64, N> {
    type Output = SVector<f64, N>;

    fn add(self, rhs: &OutOfMode<N>) -> Self::Output {
        self + rhs.0
    }
}
impl<const N: usize> Add<&SVector<f64, N>> for &WithinMode<N> {
    type Output = SVector<f64, N>;

    fn add(self, rhs: &SVector<f64, N>) -> Self::Output {
        self.0 + rhs
    }
}

impl<const N: usize> Add<&OutOfMode<N>> for &OutOfMode<N> {
    type Output = SVector<f64, N>;

    fn add(self, rhs: &OutOfMode<N>) -> Self::Output {
        self.0 + rhs.0
    }
}
impl<const N: usize> Add<&WithinMode<N>> for &OutOfMode<N> {
    type Output = SVector<f64, N>;

    fn add(self, rhs: &WithinMode<N>) -> Self::Output {
        self.0 + rhs.0
    }
}
impl<const N: usize> Add<&SVector<f64, N>> for &OutOfMode<N> {
    type Output = SVector<f64, N>;

    fn add(self, rhs: &SVector<f64, N>) -> Self::Output {
        self.0 + rhs
    }
}

// ## Subtraction
impl<const N: usize> Sub<&WithinMode<N>> for &WithinMode<N> {
    type Output = SVector<f64, N>;

    fn sub(self, rhs: &WithinMode<N>) -> Self::Output {
        self.0 - rhs.0
    }
}
impl<const N: usize> Sub<&OutOfMode<N>> for &WithinMode<N> {
    type Output = SVector<f64, N>;

    fn sub(self, rhs: &OutOfMode<N>) -> Self::Output {
        self.0 - rhs.0
    }
}
impl<const N: usize> Sub<&SVector<f64, N>> for &WithinMode<N> {
    type Output = SVector<f64, N>;

    fn sub(self, rhs: &SVector<f64, N>) -> Self::Output {
        self.0 - rhs
    }
}
impl<const N: usize> Sub<&WithinMode<N>> for &SVector<f64, N> {
    type Output = SVector<f64, N>;

    fn sub(self, rhs: &WithinMode<N>) -> Self::Output {
        self - rhs.0
    }
}
impl<const N: usize> Sub<&OutOfMode<N>> for &SVector<f64, N> {
    type Output = SVector<f64, N>;

    fn sub(self, rhs: &OutOfMode<N>) -> Self::Output {
        self - rhs.0
    }
}
impl<const N: usize> Sub<&OutOfMode<N>> for &OutOfMode<N> {
    type Output = SVector<f64, N>;

    fn sub(self, rhs: &OutOfMode<N>) -> Self::Output {
        self.0 - rhs.0
    }
}
impl<const N: usize> Sub<&WithinMode<N>> for &OutOfMode<N> {
    type Output = SVector<f64, N>;

    fn sub(self, rhs: &WithinMode<N>) -> Self::Output {
        self.0 - rhs.0
    }
}
impl<const N: usize> Sub<&SVector<f64, N>> for &OutOfMode<N> {
    type Output = SVector<f64, N>;

    fn sub(self, rhs: &SVector<f64, N>) -> Self::Output {
        self.0 - rhs
    }
}

impl<const N: usize> Add<WithinMode<N>> for WithinMode<N> {
    type Output = SVector<f64, N>;

    fn add(self, rhs: WithinMode<N>) -> Self::Output {
        self.0 + rhs.0
    }
}
impl<const N: usize> Add<OutOfMode<N>> for WithinMode<N> {
    type Output = SVector<f64, N>;

    fn add(self, rhs: OutOfMode<N>) -> Self::Output {
        self.0 + rhs.0
    }
}
impl<const N: usize> Add<SVector<f64, N>> for WithinMode<N> {
    type Output = SVector<f64, N>;

    fn add(self, rhs: SVector<f64, N>) -> Self::Output {
        self.0 + rhs
    }
}

impl<const N: usize> Add<OutOfMode<N>> for OutOfMode<N> {
    type Output = SVector<f64, N>;

    fn add(self, rhs: OutOfMode<N>) -> Self::Output {
        self.0 + rhs.0
    }
}
impl<const N: usize> Add<WithinMode<N>> for OutOfMode<N> {
    type Output = SVector<f64, N>;

    fn add(self, rhs: WithinMode<N>) -> Self::Output {
        self.0 + rhs.0
    }
}
impl<const N: usize> Add<SVector<f64, N>> for OutOfMode<N> {
    type Output = SVector<f64, N>;

    fn add(self, rhs: SVector<f64, N>) -> Self::Output {
        self.0 + rhs
    }
}

impl<const N: usize> Sub<WithinMode<N>> for WithinMode<N> {
    type Output = SVector<f64, N>;

    fn sub(self, rhs: WithinMode<N>) -> Self::Output {
        self.0 - rhs.0
    }
}
impl<const N: usize> Sub<OutOfMode<N>> for WithinMode<N> {
    type Output = SVector<f64, N>;

    fn sub(self, rhs: OutOfMode<N>) -> Self::Output {
        self.0 - rhs.0
    }
}
impl<const N: usize> Sub<SVector<f64, N>> for WithinMode<N> {
    type Output = SVector<f64, N>;

    fn sub(self, rhs: SVector<f64, N>) -> Self::Output {
        self.0 - rhs
    }
}

impl<const N: usize> Sub<OutOfMode<N>> for OutOfMode<N> {
    type Output = SVector<f64, N>;

    fn sub(self, rhs: OutOfMode<N>) -> Self::Output {
        self.0 - rhs.0
    }
}
impl<const N: usize> Sub<WithinMode<N>> for OutOfMode<N> {
    type Output = SVector<f64, N>;

    fn sub(self, rhs: WithinMode<N>) -> Self::Output {
        self.0 - rhs.0
    }
}
impl<const N: usize> Sub<SVector<f64, N>> for OutOfMode<N> {
    type Output = SVector<f64, N>;

    fn sub(self, rhs: SVector<f64, N>) -> Self::Output {
        self.0 - rhs
    }
}
