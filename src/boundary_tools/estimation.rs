use nalgebra::{Const, OMatrix, SVector};

use crate::prelude::{
    Adherer, AdhererFactory, AdhererState, Classifier, Halfspace, MeshExplorer, Result, Sample,
};

/// Given an initial halfspace, determines a more accurate surface direction and
/// returns the updated halfspace,.
/// ## Arguments
/// * d : The distance to sample from @hs.
/// * hs : The initial halfspace to improve OSV accuracy for.
/// * adherer_f : The AdhererFactory to use for finding neighboring halfspaces.
/// * classifier : The classifier for the FUT being tested.
/// ## Return (Ok((new_hs, neighbors, non_b_samples)))
/// * new_hs : The updated @hs with an improved OSV approximation.
/// * neighbors : The boundary points neighboring @hs.
/// * all_samples : All samples that were taken during the process.
/// ## Error (Err)
/// * SamplingError : If the sample is out of bounds or the boundary is lost, this
///   error can be returned. BLEs can sometimes be remedied by decreasing @hs's
///   distance from the boundary. Out of Bounds errors are due to limitations of the
///   input domain, so reducing @d's size can circumvent these issues.
pub fn approx_surface<const N: usize, F, C>(
    d: f64,
    hs: Halfspace<N>,
    adherer_f: &mut F,
    classifier: &mut C,
) -> Result<(Halfspace<N>, Vec<Halfspace<N>>, Vec<Sample<N>>)>
where
    F: AdhererFactory<N>,
    C: Classifier<N>,
{
    // Find cardinal vectors of surface
    let basis_vectors = OMatrix::<f64, Const<N>, Const<N>>::identity();
    let cardinals: Vec<SVector<f64, N>> =
        MeshExplorer::<N, F>::create_cardinals(hs.n, basis_vectors);

    let mut all_samples = vec![];

    // Find neighboring boundary points
    let mut neighbors = vec![];
    for cardinal in cardinals {
        let mut adh = adherer_f.adhere_from(hs, d * cardinal);
        loop {
            match adh.get_state() {
                AdhererState::Searching => {
                    all_samples.push(*adh.sample_next(classifier)?);
                }
                AdhererState::FoundBoundary(halfspace) => {
                    neighbors.push(halfspace);
                    break;
                }
            }
        }
    }

    // Average neighboring boundary point OSVs
    let mut new_n = SVector::zeros();
    let mut count = 0.0;
    for other_hs in neighbors.iter() {
        new_n += other_hs.n;
        count += 1.0;
    }

    new_n /= count;

    Ok((Halfspace { b: hs.b, n: new_n }, neighbors, all_samples))
}

#[cfg(test)]
mod approx_surface {
    use std::f64::consts::PI;

    use nalgebra::SVector;

    use crate::{
        prelude::{ConstantAdhererFactory, Domain, Halfspace, WithinMode},
        sps::Sphere,
    };

    use super::approx_surface;

    const RADIUS: f64 = 0.25;
    const JUMP_DIST: f64 = 0.05;

    fn get_center<const N: usize>() -> SVector<f64, N> {
        SVector::from_fn(|_, _| 0.5)
    }

    fn get_perfect_hs<const N: usize>() -> Halfspace<N> {
        let b = SVector::from_fn(|i, _| {
            if i == 0 {
                0.5 + RADIUS - JUMP_DIST * 0.25
            } else {
                0.5
            }
        });

        let n = SVector::from_fn(|i, _| if i == 0 { 1.0 } else { 0.0 });

        Halfspace {
            b: WithinMode(b),
            n,
        }
    }

    fn get_imperfect_hs<const N: usize>() -> Halfspace<N> {
        let b = SVector::from_fn(|i, _| {
            if i == 0 {
                0.5 + RADIUS - JUMP_DIST * 0.25
            } else {
                0.5
            }
        });

        let n = SVector::<f64, N>::from_fn(|_, _| 0.5).normalize();

        Halfspace {
            b: WithinMode(b),
            n,
        }
    }

    #[test]
    fn improves_imperfect_hs() {
        let domain = Domain::<10>::normalized();
        let mut sphere = Sphere::new(get_center(), RADIUS, Some(domain));

        let hs = get_imperfect_hs();

        let mut adh_f = ConstantAdhererFactory::new(5.0f64.to_radians(), None);

        let (new_hs, _, _) = approx_surface(JUMP_DIST, hs, &mut adh_f, &mut sphere)
            .expect("Unexpected sampling error");

        let correct_hs = get_perfect_hs();

        let angle = new_hs.n.angle(&correct_hs.n);

        let err = angle / PI;
        let prev_err = get_imperfect_hs::<10>().n.angle(&correct_hs.n);
        assert!(
            err <= prev_err,
            "Did not decrease OSV error. Original error of {prev_err} and got new error of {err}"
        );
    }
}
