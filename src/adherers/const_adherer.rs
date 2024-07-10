use crate::{
    adherer_core::{Adherer, AdhererFactory, AdhererState, SamplingError},
    structs::{Classifier, Halfspace, PointNode, Span},
};
use nalgebra::{Const, OMatrix, SVector};
use std::f64::consts::PI;

pub struct ConstantAdherer<const N: usize> {
    span: Span<N>,
    pivot: Halfspace<N>,
    v: SVector<f64, N>,
    samples: Vec<PointNode<N>>,
    rot: Option<OMatrix<f64, Const<N>, Const<N>>>,
    angle: f64,
    delta_angle: f64,
    max_rotation: f64,
    pub state: AdhererState<N>,
}

pub struct ConstantAdhererFactory<const N: usize> {
    delta_angle: f64,
    max_rotation: Option<f64>,
}

impl<const N: usize> ConstantAdherer<N> {
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

    fn take_initial_sample(
        &mut self,
        classifier: &Box<dyn Classifier<N>>,
    ) -> Result<PointNode<N>, SamplingError<N>> {
        let cur = self.pivot.b + self.v;
        let cls = classifier.classify(cur)?;
        let delta_angle = if cls {
            self.delta_angle
        } else {
            -self.delta_angle
        };
        self.rot = Some(self.span.get_rotater()(delta_angle));
        Ok(PointNode { p: cur, class: cls })
    }

    fn take_sample(
        &mut self,
        rot: OMatrix<f64, Const<N>, Const<N>>,
        classifier: &Box<dyn Classifier<N>>,
    ) -> Result<PointNode<N>, SamplingError<N>> {
        self.v = rot * self.v;
        let cur = self.pivot.b + self.v;
        let cls = classifier.classify(cur)?;

        self.angle += self.delta_angle;

        Ok(PointNode { p: cur, class: cls })
    }
}

impl<const N: usize> Adherer<N> for ConstantAdherer<N> {
    fn get_state(&self) -> AdhererState<N> {
        self.state
    }

    fn sample_next(
        &mut self,
        classifier: &Box<dyn Classifier<N>>,
    ) -> Result<PointNode<N>, SamplingError<N>> {
        let cur;
        let cls;
        if let Some(rot) = self.rot {
            PointNode { p: cur, class: cls } = self.take_sample(rot, classifier)?;
        } else {
            PointNode { p: cur, class: cls } = self.take_initial_sample(classifier)?;
        }

        if let Some(&PointNode {
            p: prev,
            class: prev_cls,
        }) = self.samples.last()
        {
            if cls != prev_cls {
                let b = if cls { cur } else { prev };
                let s = b - self.pivot.b;
                let rot90 = self.span.get_rotater()(PI / 2.0);
                let n = (rot90 * s).normalize();
                self.state = AdhererState::FoundBoundary(Halfspace { b, n })
            }
        }

        if matches!(self.state, AdhererState::Searching {}) && self.angle > self.max_rotation {
            return Err(SamplingError::BoundaryLost(
                cur,
                "Max rotation exceeded".to_string(),
            ));
        }

        self.samples.push(PointNode { p: cur, class: cls });

        Ok(PointNode { p: cur, class: cls })
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
    fn adhere_from(&self, hs: Halfspace<N>, v: SVector<f64, N>) -> Box<dyn Adherer<N>> {
        Box::new(ConstantAdherer::new(
            hs,
            v,
            self.delta_angle,
            self.max_rotation,
        ))
    }
}
