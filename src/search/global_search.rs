use nalgebra::SVector;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

use crate::{
    extensions::Queue,
    prelude::{BoundaryPair, Classifier, OutOfMode, Result, Sample, WithinMode},
    structs::Domain,
};

/// Random exploration of the search domain.
pub struct MonteCarloSampler<const N: usize> {
    rng: ChaCha20Rng,
    domain: Domain<N>,
}

/// A system that produces points to be sampled for the purpose of exploring a
/// domain, also referred to as global search.
pub trait DomainSampler<const N: usize> {
    fn next(&mut self) -> SVector<f64, N>;
    fn domain(&self) -> &Domain<N>;
}

/// A system for finding boundary pairs. These
pub trait GlobalSearch<const N: usize, C: Classifier<N>> {
    fn step(&mut self, classifier: &mut C) -> Result<Sample<N>>;
    fn domain(&self) -> &Domain<N>;
 
    fn pop(&mut self) -> Option<BoundaryPair<N>>;
}

impl<const N: usize> MonteCarloSampler<N> {
    pub fn new(domain: Domain<N>, seed: u64) -> Self {
        let rng = ChaCha20Rng::seed_from_u64(seed);
        MonteCarloSampler { rng, domain }
    }
}

impl<const N: usize> DomainSampler<N> for MonteCarloSampler<N> {
    fn next(&mut self) -> SVector<f64, N> {
        let v: SVector<f64, N> = SVector::from_fn(|_, _| self.rng.gen());
        v.component_mul(&self.domain.dimensions()) + self.domain.low()
    }

    fn domain(&self) -> &Domain<N> {
        &self.domain
    }
}

pub struct StandardGS<const N: usize, Ds: DomainSampler<N>> {
    domain: Domain<N>,
    sampler: Ds,
    queue: Vec<BoundaryPair<N>>,

    t: Option<WithinMode<N>>,
    x: Option<OutOfMode<N>>,
}

impl<const N: usize, C: Classifier<N>, Ds: DomainSampler<N>> GlobalSearch<N, C>
    for StandardGS<N, Ds>
{
    fn step(&mut self, classifier: &mut C) -> Result<Sample<N>> {
        let s = classifier.classify(&self.sampler.next())?;
        match s {
            Sample::WithinMode(t) => {
                if self.t.is_none() {
                    if let Some(x) = self.x {
                        self.queue.enqueue(BoundaryPair::new(t, x));
                    } else {
                        self.t = Some(t);
                    }
                }
            }
            Sample::OutOfMode(x) => {
                if self.x.is_none() {
                    if let Some(t) = self.t {
                        self.queue.enqueue(BoundaryPair::new(t, x));
                    } else {
                        self.x = Some(x);
                    }
                }
            }
        }

        Ok(s)
    }

    fn domain(&self) -> &Domain<N> {
        self.sampler.domain()
    }

    fn pop(&mut self) -> Option<BoundaryPair<N>> {
        self.queue.dequeue()
    }
}

#[cfg(test)]
mod test_monte_carlo {
    use crate::structs::Domain;

    use super::{DomainSampler, MonteCarloSampler};

    #[test]
    fn all_points_fall_in_domain() {
        //
        let domain = Domain::<10>::normalized();
        let mut mc = MonteCarloSampler::new(domain.clone(), 1);

        assert!(
            (0..10000).all(|_| domain.contains(&mc.next())),
            "MonteCarlo resulted in invalid samples - out of bounds?"
        )
    }
}
