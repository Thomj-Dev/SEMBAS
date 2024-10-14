pub mod boundary;
pub mod error;
pub mod report;
pub mod sampling;

pub use boundary::*;
pub use error::*;
pub use sampling::*;

use core::fmt;

use nalgebra::{Const, OMatrix, SVector};

use crate::utils::vector_to_string;

/// A 2-dimensional subspace of an N-dimensional input space, described by two
/// orthonormal vectors, u and v.
#[derive(Debug, Clone, PartialEq)]
pub struct Span<const N: usize> {
    u: SVector<f64, N>,
    v: SVector<f64, N>,
}

/// An N-dimensional hyperrectangle that is defined by an lower and upper bound (low
/// and high). Used to define a valid input region to sample from for a system under
/// test.
#[derive(Debug, Clone, PartialEq)]
pub struct Domain<const N: usize> {
    low: SVector<f64, N>,
    high: SVector<f64, N>,
}

impl<const N: usize> Span<N> {
    /// Constructs a Span across @u and @v. @u and @v are orthonormalized, where @v
    /// is forced to be orthogonal to @u, and @u retains its directionality. Uses
    /// Gramm Schmidt Orthonormalization
    pub fn new(u: SVector<f64, N>, v: SVector<f64, N>) -> Self {
        let u = u.normalize();
        let v = v.normalize();
        let v = (v - u * u.dot(&v)).normalize();
        Span { u, v }
    }

    pub fn u(&self) -> SVector<f64, N> {
        self.u
    }
    pub fn v(&self) -> SVector<f64, N> {
        self.v
    }

    // Provides a rotater function rot(angle: f64) which returns a rotation matrix
    // that rotates by an angle in radians along &self's span.
    pub fn get_rotater(&self) -> impl Fn(f64) -> Box<OMatrix<f64, Const<N>, Const<N>>> {
        let identity = OMatrix::<f64, Const<N>, Const<N>>::identity();

        let a = self.u * self.v.transpose() - self.v * self.u.transpose();
        let b = self.v * self.v.transpose() + self.u * self.u.transpose();

        move |angle: f64| identity + a * angle.sin() + b * (angle.cos() - 1.0)
    }
}

impl<const N: usize> Domain<N> {
    /// Returns a domain bounded by the two points.
    pub fn new(p1: SVector<f64, N>, p2: SVector<f64, N>) -> Self {
        let low = p1.zip_map(&p2, |a, b| a.min(b));
        let high = p1.zip_map(&p2, |a, b| a.max(b));

        Domain { low, high }
    }

    /// Returns a domain with the provided bounds.
    /// ## Safety
    /// This function is unsafe because it doesn't do any checks to ensure that for
    /// all dimensions, low < high. If this condition is not met, the Domain's
    /// operations behavior is undefined.
    pub unsafe fn new_from_bounds(low: SVector<f64, N>, high: SVector<f64, N>) -> Self {
        Domain { low, high }
    }

    /// Returns a Domain bounded between 0 and 1 for all dimensions.
    pub fn normalized() -> Self {
        let low = SVector::<f64, N>::zeros();
        let high = SVector::<f64, N>::repeat(1.0);
        Domain { low, high }
    }

    /// The lower bound of the domain.
    pub fn low(&self) -> &SVector<f64, N> {
        &self.low
    }

    /// The upper bound of the domain.
    pub fn high(&self) -> &SVector<f64, N> {
        &self.high
    }

    /// Checks if the given vector is within the domain.
    pub fn contains(&self, p: &SVector<f64, N>) -> bool {
        let below_low = SVector::<bool, N>::from_fn(|i, _| p[i] < self.low[i]);
        if below_low.iter().any(|&x| x) {
            return false;
        }

        let above_high = SVector::<bool, N>::from_fn(|i, _| p[i] > self.high[i]);
        if above_high.iter().any(|&x| x) {
            return false;
        }

        true
    }

    /// Returns the size of each dimension as a vector.
    pub fn dimensions(&self) -> SVector<f64, N> {
        self.high - self.low
    }

    /// Projects a point from one domain to another.
    /// Retains the relative position for all points within the source domain.
    /// Useful for projecting an input from one domain to a normalized domain and vis
    /// versa.
    /// ## Arguments
    /// * p: The point that is being projected
    /// * from: The domain that the point is projecting from
    /// * to: The domain that the point is projecting to
    pub fn project_point_domains(
        p: &SVector<f64, N>,
        from: &Domain<N>,
        to: &Domain<N>,
    ) -> SVector<f64, N> {
        ((p - from.low).component_div(&from.dimensions())).component_mul(&to.dimensions()) + to.low
    }

    /// Finds the distance between the edge of the domain from a point in the
    /// direction of the provided vector. Useful for finding target/non-target
    /// samples on the extremes of the input space.
    /// * p: A point that the ray starts from
    /// * v: The direction the ray travels
    ////// ## Returns
    /// * t: The linear distance between p and the edge of the domain in the
    ///   direction v
    pub fn distance_to_edge(&self, p: &SVector<f64, N>, v: &SVector<f64, N>) -> Result<f64> {
        let t_lower = (self.low - p).component_div(v);
        let t_upper = (self.high - p).component_div(v);

        let l = t_lower
            .iter()
            .filter(|&&xi| xi >= 0.0)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .cloned();

        let u = t_upper
            .iter()
            .filter(|&&xi| xi >= 0.0)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .cloned();

        let t = match (l, u) {
            (None, Some(t)) => t,
            (Some(t), None) => t,
            (Some(tl), Some(tu)) => tl.min(tu),
            // OOB due to point falling outside of domain
            (None, None) => return Err(SamplingError::OutOfBounds),
        };

        Ok(t)
    }
}

impl<const N: usize> fmt::Display for Span<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Span({:?}, {:?})",
            vector_to_string(&self.u),
            vector_to_string(&self.v)
        )
    }
}

#[cfg(test)]
mod span_tests {
    use super::*;

    const ATOL: f64 = 1e-10;

    fn approx(a: f64, b: f64, atol: f64) -> bool {
        (a - b).abs() < atol
    }

    #[test]
    fn is_orthogonal() {
        let u = nalgebra::vector![0.5, 0.5, 0.1, 0.4, 1.0];
        let v = nalgebra::vector![1.1, -0.2, -0.5, 0.1, 0.8];

        let span = Span::new(u, v);

        assert!(approx(span.u.angle(&span.v).to_degrees(), 90.0, ATOL));
    }

    #[test]
    fn is_normal() {
        let u = nalgebra::vector![0.5, 0.5, 0.1, 0.4, 1.0];
        let v = nalgebra::vector![1.1, -0.2, -0.5, 0.1, 0.8];

        let span = Span::new(u, v);

        assert!(approx(span.u.norm(), 1.0, ATOL));
        assert!(approx(span.v.norm(), 1.0, ATOL));
    }

    #[test]
    fn rotater_90() {
        let u = nalgebra::vector![0.5, 0.5, 0.1, 0.4, 1.0];
        let v = nalgebra::vector![1.1, -0.2, -0.5, 0.1, 0.8];

        let span = Span::new(u, v);

        let x0 = v;
        let angle = 90.0f64.to_radians();

        let x1 = span.get_rotater()(angle) * x0;

        assert!(approx(x0.angle(&x1), angle, ATOL));
    }

    #[test]
    fn rotater_25() {
        let u = nalgebra::vector![0.5, 0.5, 0.1, 0.4, 1.0];
        let v = nalgebra::vector![1.1, -0.2, -0.5, 0.1, 0.8];

        let span = Span::new(u, v);

        let x0 = v;
        let angle = 25.0f64.to_radians();

        let x1 = span.get_rotater()(angle) * x0;

        assert!(approx(x0.angle(&x1), angle, ATOL));
    }
}

#[cfg(test)]
mod domain_tests {
    use nalgebra::vector;

    use super::*;

    const ATOL: f64 = 1e-10;

    fn is_near<const N: usize>(a: &SVector<f64, N>, b: &SVector<f64, N>, atol: f64) -> bool {
        (b - a).norm() <= atol
    }

    #[test]
    fn point_translation_low_to_low() {
        let src = Domain::<3>::normalized();
        let dst = Domain::<3>::new(vector![1.0, 2.5, 3.5], vector![4.0, 5.0, 6.0]);

        let p0 = src.low();
        let p1 = Domain::project_point_domains(p0, &src, &dst);

        assert!(is_near(&p1, dst.low(), ATOL))
    }

    #[test]
    fn point_translation_high_to_high() {
        let src = Domain::<3>::normalized();
        let dst = Domain::<3>::new(vector![1.0, 2.5, 3.5], vector![4.0, 5.0, 6.0]);

        let p0 = src.high();
        let p1 = Domain::project_point_domains(p0, &src, &dst);

        assert!(is_near(&p1, dst.high(), ATOL))
    }

    #[test]
    fn point_translation_mid_to_mid() {
        let src = Domain::<3>::normalized();
        let dst = Domain::<3>::new(vector![1.0, 2.5, 3.5], vector![4.0, 5.0, 6.0]);

        let src_mid = src.low() + src.dimensions() / 2.0;
        let dst_mid = dst.low() + dst.dimensions() / 2.0;

        let p0 = src_mid;
        let p1 = Domain::project_point_domains(&p0, &src, &dst);

        assert!(is_near(&p1, &dst_mid, ATOL))
    }

    #[test]
    fn low_is_below_high() {
        let d = Domain::<3>::new(vector![4.0, 2.5, 6.0], vector![1.0, 5.0, 3.5]);

        assert!(d.low().iter().zip(d.high.iter()).all(|(l, h)| l < h));
    }

    #[test]
    fn contains_false_when_below_low() {
        let d = Domain::<3>::new(vector![4.0, 2.5, 6.0], vector![1.0, 5.0, 3.5]);
        let p = d.low() - vector![0.01, 0.01, 0.01];

        assert!(!d.contains(&p))
    }

    #[test]
    fn contains_true_when_on_low() {
        let d = Domain::<3>::new(vector![4.0, 2.5, 6.0], vector![1.0, 5.0, 3.5]);
        let p = d.low();

        assert!(d.contains(p))
    }

    #[test]
    fn contains_false_when_above_high() {
        let d = Domain::<3>::new(vector![4.0, 2.5, 6.0], vector![1.0, 5.0, 3.5]);
        let p = d.high() + vector![0.01, 0.01, 0.01];

        assert!(!d.contains(&p))
    }

    #[test]
    fn contains_true_when_on_high() {
        let d = Domain::<3>::new(vector![4.0, 2.5, 6.0], vector![1.0, 5.0, 3.5]);
        let p = d.high();

        assert!(d.contains(p))
    }
}
