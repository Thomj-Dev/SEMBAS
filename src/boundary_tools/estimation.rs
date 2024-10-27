use nalgebra::{Const, OMatrix, SVector};

use crate::{
    prelude::{
        Adherer, AdhererFactory, AdhererState, Boundary, BoundaryRTree, Classifier, Domain,
        Halfspace, MeshExplorer, Result, Sample,
    },
    search::global_search::{DomainSampler, MonteCarloSampler},
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

/// Predicts whether or not some point, @p, will be classified as WithinMode or
/// OutOfMode according to the explored boundary. As a result, does not require the
/// classifier for the fut.
/// ## Arguments
/// * p : The point to be classified.
/// * boundary : The explored boundary for the target performance mode.
/// * btree : The RTree for @boundary.
/// * k : The number of halfspaces to consider while classifier @p. A good default is
///   1, but with higher resolution and dimensional boundaries, playing with this
///   number may improve results.
pub fn approx_prediction<const N: usize>(
    p: SVector<f64, N>,
    boundary: &Boundary<N>,
    btree: &BoundaryRTree<N>,
    k: u32,
) -> Sample<N> {
    let mut cls = true;
    for (_, neighbor) in (0..k).zip(btree.nearest_neighbor_iter(&p.into())) {
        let hs = boundary.get(neighbor.data).expect(
            "Invalid neighbor index used on @boundary. Often a result of @boundary being out of sync or entirely different from @btree."
        );

        let s = (p - *hs.b).normalize();
        if s.dot(&hs.n) > 0.0 {
            cls = false;
            break;
        }
    }

    Sample::from_class(p, cls)
}

/// Estimates the volume of an envelope using Monte Carlo sampling using approximate
/// predictions.
/// ## Arguments
/// * boundary : The boundary of the envelope whose volume is being measured.
/// * btree : The RTree for the boundary.
/// * n_samples : How many samples to take for estimating volume. More -> higher
///   accuracy
/// * n_neighbors : Varies how many halfspaces should be considered while determining
///   if a point falls within an envelope. A good default is 1, but with higher
///   resolution and dimensional boundaries playing with this number may improve
///   results.
/// * seed : The seed to use while generating random points for MC.
/// ## Return
/// * volume : The volume that lies within the envelope.
pub fn approx_mc_volume<const N: usize>(
    boundary: &Boundary<N>,
    btree: &BoundaryRTree<N>,
    n_samples: u32,
    n_neighbors: u32,
    seed: u64,
) -> f64 {
    let point_cloud: Vec<_> = boundary.iter().map(|hs| *hs.b).collect();
    let domain = Domain::new_from_point_cloud(&point_cloud);

    let mut mc = MonteCarloSampler::new(domain, seed);

    let mut wm_count = 0;

    for _ in 0..n_samples {
        if approx_prediction(mc.sample(), boundary, btree, n_neighbors).class() {
            wm_count += 1;
        }
    }

    let ratio = wm_count as f64 / n_samples as f64;

    ratio * mc.domain().volume()
}

/// Estimates the volume of an envelope using Monte Carlo sampling using approximate
/// predictions.
/// ## Arguments
/// * b1 : The first boundary.
/// * b2 : The second boundary.
/// * btree1 : The RTree for the first boundary.
/// * btree : The RTree for the second boundary.
/// * n_samples : How many samples to take for estimating volume. More -> higher
///   accuracy
/// * n_neighbors : Varies how many halfspaces should be considered while determining
///   if a point falls within an envelope. A good default is 1, but with higher
///   resolution and dimensional boundaries playing with this number may improve
///   results.
/// * seed : The seed to use while generating random points for MC.
/// ## Return (intersection_volume, envelope1_volume, envelope2_volume)
/// * intersection_volume : The volume that lies in both envelope 1 and 2.
/// * envelope1_volume : The volume that lies only with envelope1.
/// * envelope2_volume : The volume that lies only with envelope2.
///
/// The total volume is the sum of these voumes. The total volume of an envelop is
/// the sum of its volume and the intersection volume.
pub fn approx_mc_volume_intersection<const N: usize>(
    b1: &Boundary<N>,
    b2: &Boundary<N>,
    btree1: &BoundaryRTree<N>,
    btree2: &BoundaryRTree<N>,
    n_samples: u32,
    n_neighbors: u32,
    seed: u64,
) -> (f64, f64, f64) {
    let pc: Vec<_> = b1.iter().chain(b2).map(|hs| *hs.b).collect();

    let mut mc = MonteCarloSampler::new(Domain::new_from_point_cloud(&pc), seed);

    let mut b1_only_count = 0;
    let mut b2_only_count = 0;
    let mut both_count = 0;

    for _ in 0..n_samples {
        let p = mc.sample();
        let cls1 = approx_prediction(p, b1, btree1, n_neighbors).class();
        let cls2 = approx_prediction(p, b2, btree2, n_neighbors).class();

        if cls1 && cls2 {
            both_count += 1;
        } else if cls1 {
            b1_only_count += 1;
        } else if cls2 {
            b2_only_count += 1;
        }
    }

    let ratio1 = b1_only_count as f64 / n_samples as f64;
    let ratio2 = b2_only_count as f64 / n_samples as f64;
    let intersect_ratio = both_count as f64 / n_samples as f64;

    let vol = mc.domain().volume();
    let b1_vol = ratio1 * vol;
    let b2_vol = ratio2 * vol;
    let intesect_vol = intersect_ratio * vol;

    (intesect_vol, b1_vol, b2_vol)
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
