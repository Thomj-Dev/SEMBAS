use nalgebra::SVector;

use crate::{
    prelude::{Result, Sample},
    structs::{Classifier, Domain},
};

pub struct Sphere<const N: usize> {
    center: SVector<f64, N>,
    radius: f64,
    domain: Option<Domain<N>>,
}

impl<const N: usize> Sphere<N> {
    pub fn new(center: SVector<f64, N>, radius: f64, domain: Option<Domain<N>>) -> Sphere<N> {
        Sphere {
            center,
            radius,
            domain,
        }
    }

    pub fn center(&self) -> &SVector<f64, N> {
        &self.center
    }

    pub fn radius(&self) -> f64 {
        self.radius
    }

    pub fn domain(&self) -> Option<&Domain<N>> {
        self.domain.as_ref()
    }
}

impl<const N: usize> Classifier<N> for Sphere<N> {
    fn classify(&mut self, p: SVector<f64, N>) -> Result<Sample<N>> {
        if let Some(domain) = &self.domain {
            if !domain.contains(&p) {
                return Err(crate::structs::SamplingError::OutOfBounds);
            }
        }

        Ok(Sample::from_class(
            p,
            (self.center - p).norm() <= self.radius,
        ))
    }
}

pub struct Cube<const N: usize> {
    shape: Domain<N>,
    domain: Option<Domain<N>>,
}

impl<const N: usize> Cube<N> {
    pub fn new(shape: Domain<N>, domain: Option<Domain<N>>) -> Cube<N> {
        Cube { shape, domain }
    }

    pub fn from_size(size: f64, center: SVector<f64, N>, domain: Option<Domain<N>>) -> Self {
        let low = center - SVector::from_fn(|_, _| size / 2.0);
        let high = center + SVector::from_fn(|_, _| size / 2.0);
        let shape = Domain::new(low, high);
        Cube { shape, domain }
    }

    pub fn shape(&self) -> &Domain<N> {
        &self.shape
    }

    pub fn domain(&self) -> Option<&Domain<N>> {
        self.domain.as_ref()
    }
}

impl<const N: usize> Classifier<N> for Cube<N> {
    fn classify(&mut self, p: SVector<f64, N>) -> Result<Sample<N>> {
        if let Some(domain) = &self.domain {
            if !domain.contains(&p) {
                return Err(crate::structs::SamplingError::OutOfBounds);
            }
        }

        Ok(Sample::from_class(p, self.shape.contains(&p)))
    }
}

pub struct SphereCluster<const N: usize> {
    spheres: Vec<Sphere<N>>,
    domain: Option<Domain<N>>,
}

impl<const N: usize> SphereCluster<N> {
    pub fn new(spheres: Vec<Sphere<N>>, domain: Option<Domain<N>>) -> Self {
        SphereCluster { spheres, domain }
    }

    pub fn spheres(&self) -> &[Sphere<N>] {
        &self.spheres
    }

    pub fn domain(&self) -> Option<&Domain<N>> {
        self.domain.as_ref()
    }
}

impl<const N: usize> Classifier<N> for SphereCluster<N> {
    fn classify(&mut self, p: SVector<f64, N>) -> Result<Sample<N>> {
        if let Some(domain) = &self.domain {
            if !domain.contains(&p) {
                return Err(crate::structs::SamplingError::OutOfBounds);
            }
        }

        for sphere in self.spheres.iter_mut() {
            if let Sample::WithinMode(t) = sphere.classify(p)? {
                return Ok(Sample::WithinMode(t));
            };
        }

        Ok(Sample::from_class(p, false))
    }
}
