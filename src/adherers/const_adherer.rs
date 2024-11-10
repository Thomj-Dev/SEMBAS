use crate::{
    adherer_core::{Adherer, AdhererFactory, AdhererState},
    structs::{Classifier, Halfspace, Result, Sample, SamplingError, Span},
};
use nalgebra::{Const, OMatrix, SVector};
#[cfg(feature = "io")]
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Pivots around a known boundary halfspace by taking fixed-angle rotations until
/// the boundary is crossed.
#[derive(Debug)]
pub struct ConstantAdherer<const N: usize> {
    span: Span<N>,
    pivot: Halfspace<N>,
    v: SVector<f64, N>,
    samples: Vec<Sample<N>>,
    rot: Option<OMatrix<f64, Const<N>, Const<N>>>,
    angle: f64,
    delta_angle: f64,
    max_rotation: f64,
    pub state: AdhererState<N>,
}

/// Builds a ConstantAdherer instance.
#[cfg_attr(feature = "io", derive(Serialize, Deserialize))]
#[derive(Debug, Copy, Clone)]
pub struct ConstantAdhererFactory<const N: usize> {
    delta_angle: f64,
    max_rotation: Option<f64>,
}

impl<const N: usize> ConstantAdherer<N> {
    /// Creates a ConstantAdherer.
    /// ## Arguments
    /// * pivot : The halfspace to rotate around for finding a neighboring halfspace.
    /// * v : The vector of travel, the direction along the surface to explore in.
    ///     The length of this vector determines how far to travel.
    /// * delta_angle : The fixed-angle to rotate the displacement vector by to cross
    ///   and find the neighboring boundary.
    /// * max_rotation : The maximum total angle in radians to rotate by. Defaults to
    ///   180 degress.
    /// ## Characteristics
    /// * Boundary Sampling Efficiency (BSE): 0 <= BSE <= 1
    ///     * BSE varies with delta_angle, v.norm(), and the shape of the envelope.
    /// * Boundary Error: 0 <= err <= v.norm() * sin(delta_angle)
    ///     * Average case will be v.norm() * sin(delta_angle) / 2
    pub fn new(
        pivot: Halfspace<N>,
        v: SVector<f64, N>,
        delta_angle: f64,
        max_rotation: Option<f64>,
    ) -> Self {
        let span = Span::new(pivot.n, v);

        let max_rotation = max_rotation.unwrap_or(PI);

        ConstantAdherer {
            span,
            pivot,
            v,
            delta_angle,
            max_rotation,
            rot: None,
            samples: vec![],
            angle: 0.0,
            state: AdhererState::Searching,
        }
    }

    fn take_initial_sample<C: Classifier<N>>(&mut self, classifier: &mut C) -> Result<Sample<N>> {
        let cur = self.pivot.b + self.v;
        let sample = classifier.classify(cur)?;
        let cls = sample.class();
        let delta_angle = if cls {
            self.delta_angle
        } else {
            -self.delta_angle
        };
        self.rot = Some(self.span.get_rotater()(delta_angle));
        Ok(sample)
    }

    fn take_sample<C: Classifier<N>>(
        &mut self,
        rot: OMatrix<f64, Const<N>, Const<N>>,
        classifier: &mut C,
    ) -> Result<Sample<N>> {
        self.v = rot * self.v;
        let cur = self.pivot.b + self.v;
        self.angle += self.delta_angle;

        classifier.classify(cur)
    }
}

impl<const N: usize> Adherer<N> for ConstantAdherer<N> {
    fn get_state(&self) -> AdhererState<N> {
        self.state
    }

    fn sample_next<C: Classifier<N>>(&mut self, classifier: &mut C) -> Result<&Sample<N>> {
        let cur = if let Some(rot) = self.rot {
            self.take_sample(rot, classifier)?
        } else {
            self.take_initial_sample(classifier)?
        };

        if let Some(prev) = self.samples.last() {
            match (&cur, &prev) {
                (Sample::WithinMode(t), Sample::OutOfMode(_))
                | (Sample::OutOfMode(_), Sample::WithinMode(t)) => {
                    let b = *t;
                    let s = b - self.pivot.b;
                    let rot90 = self.span.get_rotater()(PI / 2.0);
                    let n = (rot90 * s).normalize();
                    self.state = AdhererState::FoundBoundary(Halfspace { b, n });
                }
                _ => (),
            }
        }

        if matches!(self.state, AdhererState::Searching {}) && self.angle > self.max_rotation {
            return Err(SamplingError::BoundaryLost);
        }

        self.samples.push(cur);

        Ok(self
            .samples
            .last()
            .expect("Invalid state, cur was not added to samples?"))
    }
}

impl<const N: usize> ConstantAdhererFactory<N> {
    pub fn new(delta_angle: f64, max_rotation: Option<f64>) -> Self {
        ConstantAdhererFactory {
            delta_angle,
            max_rotation,
        }
    }
}

impl<const N: usize> AdhererFactory<N> for ConstantAdhererFactory<N> {
    type TargetAdherer = ConstantAdherer<N>;
    fn adhere_from(&self, hs: Halfspace<N>, v: SVector<f64, N>) -> ConstantAdherer<N> {
        ConstantAdherer::new(hs, v, self.delta_angle, self.max_rotation)
    }
}

#[cfg(test)]
mod constant_adherer {
    use nalgebra::SVector;

    use crate::prelude::{Adherer, AdhererState, FunctionClassifier, Halfspace, WithinMode};

    use super::ConstantAdherer;

    #[test]
    fn displacement_vector_norm_never_changes() {
        let pivot = Halfspace {
            b: WithinMode(SVector::from_fn(|_, _| 0.21501)),
            n: SVector::<f64, 10>::from_fn(|i, _| if i == 0 { 1.0 } else { 0.25 }).normalize(),
        };
        let v = SVector::from_fn(|i, _| if i == 0 { 1.0 } else { 0.0 });

        let d = 0.05;
        let mut adh = ConstantAdherer::new(pivot, d * v, 5.0f64.to_radians(), None);

        let mut classifier = FunctionClassifier::new(|_| Ok(false));

        while let AdhererState::Searching = adh.get_state() {
            match adh.sample_next(&mut classifier) {
                Ok(_) => (),
                Err(_) => break,
            }

            let magnitude = adh.v.norm();
            assert!(
                (magnitude - d).abs() <= 1.0e-10,
                "Bad displacement vector! {magnitude}"
            );
        }
    }
}
