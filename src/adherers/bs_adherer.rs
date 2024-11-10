use crate::{
    adherer_core::{Adherer, AdhererFactory, AdhererState},
    structs::{Classifier, Halfspace, OutOfMode, Result, Sample, SamplingError, Span, WithinMode},
};
use nalgebra::{Const, OMatrix, SVector};
#[cfg(feature = "io")]
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Pivots around a known boundary halfspace by taking fixed-angle rotations until
/// the boundary is crossed.
pub struct BinarySearchAdherer<const N: usize> {
    pivot: Halfspace<N>,
    v: SVector<f64, N>,
    samples: Vec<Sample<N>>,
    n_iter: u32,
    angle: f64,
    prev_cls: Option<bool>,
    t: Option<WithinMode<N>>,
    x: Option<OutOfMode<N>>,
    rot_factory: Box<dyn Fn(f64) -> OMatrix<f64, Const<N>, Const<N>>>,
    pub state: AdhererState<N>,
}

/// Builds a ConstantAdherer instance.
#[cfg_attr(feature = "io", derive(Serialize, Deserialize))]
#[derive(Debug, Copy, Clone)]
pub struct BinarySearchAdhererFactory<const N: usize> {
    init_angle: f64,
    n_iter: u32,
}

impl<const N: usize> BinarySearchAdherer<N> {
    /// Creates a BinarySearchAdherer
    /// ## Arguments
    /// * pivot : The halfspace to rotate around for finding a neighboring halfspace.
    /// * v : The vector of travel, the direction along the surface to explore in.
    ///     The length of this vector determines how far to travel.
    /// * init_angle : The initial angle to rotate by. Recommended values fall
    ///   between 90 and 120 degrees.
    /// * n_iter : The number of iterations to take before returning the acquired
    ///   halfspace.
    /// ## Characteristics
    /// * Boundary Sampling Efficiency (BSE): 0 <= BSE <= 1 / (n_iter - 1)
    ///     * BSE will be equal to the upper bound unless boundary lost / out of
    ///       bounds errors occur
    /// * Boundary Error: 0 <= err <= v.norm() * sin(init_angle / 2^(n_iter - 1))
    ///     * Average case will be v.norm() * sin(init_angle / 2^(n_iter))
    pub fn new(pivot: Halfspace<N>, v: SVector<f64, N>, init_angle: f64, n_iter: u32) -> Self {
        let rot_factory = Span::new(pivot.n, v).get_rotater();

        BinarySearchAdherer {
            pivot,
            v,
            samples: vec![],
            n_iter,
            angle: init_angle,
            prev_cls: None,
            t: None,
            x: None,
            rot_factory: Box::new(rot_factory),
            state: AdhererState::Searching,
        }
    }

    fn take_initial_sample<C: Classifier<N>>(&mut self, classifier: &mut C) -> Result<Sample<N>> {
        let cur = self.pivot.b + self.v;
        let sample = classifier.classify(cur)?;
        self.prev_cls = Some(sample.class());

        match sample {
            Sample::WithinMode(t) => self.t = Some(t),
            Sample::OutOfMode(x) => self.x = Some(x),
        }

        self.n_iter -= 1;

        Ok(sample)
    }

    fn take_sample<C: Classifier<N>>(
        &mut self,
        prev_cls: bool,
        classifier: &mut C,
    ) -> Result<Sample<N>> {
        let ccw = if prev_cls { 1.0 } else { -1.0 };
        let rot = (self.rot_factory)(ccw * self.angle);
        self.v = rot * self.v;

        let cur = self.pivot.b + self.v;
        let sample = classifier.classify(cur)?;
        self.prev_cls = Some(sample.class());

        match sample {
            Sample::WithinMode(t) => self.t = Some(t),
            Sample::OutOfMode(x) => self.x = Some(x),
        }

        self.angle /= 2.0;
        self.n_iter -= 1;

        Ok(sample)
    }
}

impl<const N: usize> Adherer<N> for BinarySearchAdherer<N> {
    fn get_state(&self) -> AdhererState<N> {
        self.state
    }

    fn sample_next<C: Classifier<N>>(&mut self, classifier: &mut C) -> Result<&Sample<N>> {
        let cur = if let Some(prev_cls) = self.prev_cls {
            self.take_sample(prev_cls, classifier)?
        } else {
            self.take_initial_sample(classifier)?
        };

        if self.n_iter == 0 {
            if let (Some(t), Some(_)) = (self.t, self.x) {
                let rot90 = (self.rot_factory)(PI / 2.0);
                let b = t;
                let s = b - self.pivot.b;
                let n = (rot90 * s).normalize();
                self.state = AdhererState::FoundBoundary(Halfspace { b, n })
            } else {
                return Err(SamplingError::BoundaryLost);
            }
        }

        self.samples.push(cur);

        Ok(self
            .samples
            .last()
            .expect("Invalid state, cur was not added to samples?"))
    }
}

impl<const N: usize> BinarySearchAdhererFactory<N> {
    pub fn new(init_angle: f64, n_iter: u32) -> Self {
        BinarySearchAdhererFactory { init_angle, n_iter }
    }
}

impl<const N: usize> AdhererFactory<N> for BinarySearchAdhererFactory<N> {
    type TargetAdherer = BinarySearchAdherer<N>;
    fn adhere_from(&self, hs: Halfspace<N>, v: SVector<f64, N>) -> BinarySearchAdherer<N> {
        BinarySearchAdherer::new(hs, v, self.init_angle, self.n_iter)
    }
}
