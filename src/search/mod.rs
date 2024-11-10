use nalgebra::SVector;
use surfacing::binary_surface_search;

use crate::{
    extensions::Queue,
    structs::{BoundaryPair, Classifier, Domain, OutOfMode, Result, WithinMode},
};

#[cfg(feature = "global_search")]
pub mod global_search;

#[cfg(feature = "surfacing")]
pub mod surfacing;

/// SearchMode defines how binary search is executed.
/// * Full : Searches every recurrent mid-point between a and b.
/// * Nearest : Searches the points closer to a from the midpoint between a and b,
///   ignoring the points that fall closer to b.
///     Efficient when you are trying to re-acquire the same performance space as point
///     a, since the space beyond the first mid-point will be ignored entirely.
pub enum SearchMode {
    Full,
    Nearest,
}

/// Searches the space between two points for a specific performance mode.
/// ## Arguments
/// * mode : The search mode to use while looking for the target class.
/// * target_class : The performance mode (in-mode or out-of-mode) to search for.
///   Function termins which target class is found.
/// * max_samples : A limit on how many samples are taken before terminating. Returns
///   None if max samples is reached before finding the @target_cls.
/// * p1, p2 : The two points to search between.
/// * classifier : The FUT.
////// ## Returns
/// * Some(p) : The point that is classified as target_cls, i.e.
///   classifier.classify(p) == target_cls
/// * None : No target_cls points were found within max_samples number of iterations.
pub fn binary_search_between<const N: usize, C: Classifier<N>>(
    mode: SearchMode,
    target_cls: bool,
    max_samples: u32,
    p1: SVector<f64, N>,
    p2: SVector<f64, N>,
    classifier: &mut C,
) -> Option<SVector<f64, N>> {
    let mut pairs = vec![(p1, p2)];

    for _ in 0..max_samples {
        let (p1, p2) = pairs
            .dequeue()
            .expect("Unexpectedly ran out of pairs to explore during search?");
        let s = p2 - p1;
        let mid = p1 + s / 2.0;
        let cls = classifier
            .classify(mid)
            .expect(
                "Classifier threw error when sampling. Make sure @p1 and @p2 are valid samples?",
            )
            .class();
        if cls == target_cls {
            return Some(mid);
        }

        match mode {
            SearchMode::Full => {
                pairs.enqueue((p1, mid));
                pairs.enqueue((mid, p2));
            }
            SearchMode::Nearest => pairs.enqueue((p1, mid)),
        }
    }

    None
}

/// Finds a boundary point on the opposite side of an envelope given a starting
/// boundary point and directional vector of the chord between these boundary points.
/// **Note: If there is no Out-of-Mode samples between @b and the edge of the domain,
/// the point lying on the edge of the domain will be returned.**
/// ## Arguments
/// * max_err : The maximum allowable distance from the boundary.
///   point.
/// * b : A point that lies on the boundary of the envelope whose diameter is being
///   measured.
/// * v : A unit vector describing the direction of the chord. Note: v must point
///   TOWARDS the envelope, not away. i.e. its dot product with the OSV should be
///   negative, n.dot(v) < 0,
/// * domain : The domain to constrain exploration to be within.
/// * classifier : The classifier for the FUT.
////// ## Returns (Ok)
/// * Ok(b2) : The point within the envelope opposite that of @b.
/// ## Error (Err)
/// * Err(SamplingError) : Failed to find any target performance samples in the
///   direction @v. Often caused by an invalid v, n.dot(v) > 0, or insufficient
///   @max_samples.
/// ## Warning
/// * If @v is not facing the geometry, it may still return a boundary point with a
///   sufficiently large @num_checks or innaccurate @b. This is because it will
///   converge upon the same side of the geometry as @b.
pub fn find_opposing_boundary<const N: usize, C: Classifier<N>>(
    max_err: f64,
    t0: WithinMode<N>,
    v: SVector<f64, N>,
    domain: &Domain<N>,
    classifier: &mut C,
    num_checks: u32,
    num_iter: u32,
) -> Result<WithinMode<N>> {
    let dist = domain.distance_to_edge(&t0, &v)? * 0.999;
    let p = t0 + v * dist;

    let cls = classifier.classify(p).unwrap_or_else(|_| panic!("A point that was supposed to be on the edge of the domain (yet inside) fell outside of the classifier's domain. Incorrect @domain? p = {p:?}, v = {v:?}")).class();

    let (mut t, mut x) = if cls {
        (Some(p), None)
    } else {
        // Find next target hit
        let t = binary_search_between(SearchMode::Nearest, true, num_checks, *t0, p, classifier);

        (t, Some(p))
    };

    // While there are gaps, explore towards envelope from @b
    if let Some(mut t1) = t {
        while let Some(gap) =
            binary_search_between(SearchMode::Full, false, num_checks, t1, *t0, classifier)
        {
            x = Some(gap);
            match binary_search_between(SearchMode::Nearest, true, num_checks, *t0, gap, classifier)
            {
                Some(p) => {
                    t = Some(p);
                    t1 = p;
                }
                None => {
                    t = Some(*t0);
                    break;
                }
            };
        }
    }

    let b = match (t, x) {
        (None, None) => panic!("[Invalid state] Failed to find either a gap or any envelope. This shouldn't be possible."),
        (None, Some(_)) => t0,
        (Some(t), None) => WithinMode(t),
        (Some(t), Some(x)) => binary_surface_search(
            max_err,
            &BoundaryPair::new(WithinMode(t), OutOfMode(x)),
            num_iter,
            classifier,
        ).expect("Unexpected sampling error during final binary surface search of opposing boundary point.").b,
    };

    Ok(b)
}

#[cfg(all(test, feature = "sps"))]
mod search_tests {
    use super::*;
    use crate::{
        prelude::Sample,
        sps::Sphere,
        structs::{Classifier, Domain},
    };
    use nalgebra::SVector;

    const RADIUS: f64 = 0.25;

    struct EmptyClassifier<const N: usize> {}
    impl<const N: usize> Classifier<N> for EmptyClassifier<N> {
        fn classify(&mut self, p: SVector<f64, N>) -> Result<Sample<N>> {
            Ok(Sample::from_class(p, false))
        }
    }

    fn create_sphere<const N: usize>() -> Sphere<N> {
        let c: SVector<f64, N> = SVector::from_fn(|_, _| 0.5);

        Sphere::new(c, RADIUS, Some(Domain::normalized()))
    }
    mod binary_search_between {
        use super::*;

        #[test]
        fn finds_sphere() {
            let mut classifier = create_sphere::<10>();
            let p1: SVector<f64, 10> = SVector::zeros();
            let p2 = SVector::from_fn(|_, _| 1.0);

            let r = binary_search_between(SearchMode::Full, true, 10, p1, p2, &mut classifier)
                .expect("Failed to find sphere when it should have?");

            assert!(
                classifier
                    .classify(r)
                    .expect("Unexpected out of bounds sample from BSB result?")
                    .class(),
                "Returned non-target (incorrect) sample?"
            )
        }

        #[test]
        fn returns_none_when_no_envelope_exists() {
            let p1: SVector<f64, 10> = SVector::zeros();
            let p2 = SVector::from_fn(|_, _| 1.0);
            let mut classifier = EmptyClassifier {};

            let r = binary_search_between(SearchMode::Full, true, 10, p1, p2, &mut classifier);

            assert_eq!(r, None, "Somehow found an envelope when none existed?")
        }

        #[test]
        fn returns_none_with_insufficient_max_samples() {
            let p1: SVector<f64, 10> = SVector::zeros();
            let p2 = SVector::from_fn(|_, _| 1.0);

            let c = p2 / 8.0;
            let mut classifier = Sphere::new(c, 0.1, Some(Domain::normalized()));
            let num_steps_to_find = 4;

            let r = binary_search_between(
                SearchMode::Full,
                true,
                num_steps_to_find - 1,
                p1,
                p2,
                &mut classifier,
            );

            assert_eq!(r, None, "Found the envelope when it shouldn't have.");
        }

        #[test]
        fn finds_sphere_with_exact_max_samples() {
            let p1: SVector<f64, 10> = SVector::zeros();
            let p2 = SVector::from_fn(|_, _| 1.0);

            let c = p2 / 8.0;
            let mut classifier = Sphere::new(c, 0.1, Some(Domain::normalized()));
            let num_steps_to_find = 4;

            binary_search_between(
                SearchMode::Full,
                true,
                num_steps_to_find,
                p1,
                p2,
                &mut classifier,
            )
            .expect("Failed to find envelope with the correct max_samples.");
        }
    }

    #[cfg(test)]
    mod find_opposing_boundary {
        use crate::sps::SphereCluster;

        use super::*;

        #[test]
        fn finds_opposing_boundary_of_sphere() {
            let d = 0.01;

            let domain = Domain::normalized();
            let mut classifier = create_sphere::<10>();
            let b: SVector<f64, 10> =
                SVector::from_fn(|i, _| if i == 0 { 0.5 - RADIUS + d * 0.75 } else { 0.5 });

            let v: SVector<f64, 10> = SVector::from_fn(|i, _| if i == 0 { 1.0 } else { 0.0 });

            let b2 =
                find_opposing_boundary(0.01, WithinMode(b), v, &domain, &mut classifier, 10, 10)
                    .expect("Unexpected error on sampling a constant location?");

            assert!(
                classifier
                    .classify(b2.into())
                    .expect("Unexpected out of bounds sample for opposing boundary sample?")
                    .class(),
                "Returned non-target (incorrect) sample?"
            );

            assert!(
                ((b2 - b).magnitude() - 2.0 * RADIUS) <= 2.0 * d,
                "Resulting boundary point was not on opposite side of sphere?"
            );
        }

        #[test]
        fn returns_domain_edge_when_boundary_outside_of_domain() {
            let d = 0.01;
            let domain = Domain::normalized();

            let c = SVector::from_fn(|i, _| if i == 0 { 1.0 } else { 0.5 });
            let mut classifier = Sphere::new(c, RADIUS, Some(Domain::normalized()));

            let b: SVector<f64, 10> =
                SVector::from_fn(|i, _| if i == 0 { 1.0 - RADIUS + d * 0.75 } else { 0.5 });

            let v: SVector<f64, 10> = SVector::from_fn(|i, _| if i == 0 { 1.0 } else { 0.0 });

            let b2 =
                find_opposing_boundary(0.01, WithinMode(b), v, &domain, &mut classifier, 10, 10)
                    .expect("Unexpected error on sampling a constant location?");

            assert!(
                classifier
                    .classify(b2.into())
                    .expect("Unexpected out of bounds sample for opposing boundary sample?")
                    .class(),
                "Returned non-target (incorrect) sample?"
            );

            assert!(
                ((b2 - b).magnitude() - RADIUS) <= d,
                "Resulting boundary point was not on the domain's edge?"
            );
        }

        #[test]
        fn returns_t0_when_on_boundary_in_v_direction() {
            let d = 0.01;

            let domain = Domain::normalized();
            let mut classifier = create_sphere::<10>();
            let t0: SVector<f64, 10> = SVector::from_fn(|i, _| {
                if i == 0 {
                    0.5 - RADIUS + d * 0.001
                } else {
                    0.5
                }
            });

            let invalid_v: SVector<f64, 10> =
                -SVector::from_fn(|i, _| if i == 0 { 1.0 } else { 0.0 });

            let b = find_opposing_boundary(
                0.01,
                WithinMode(t0),
                invalid_v,
                &domain,
                &mut classifier,
                10,
                10,
            )
            .expect("Expected valid boundary but got error?");

            assert_eq!(
                (b - t0).norm(),
                0.0,
                "Expected b == t0, but found new boundary?"
            )
        }

        #[test]
        fn returns_correct_envelope_boundary_when_multiple_envelopes_exist() {
            let d = 0.01;
            let domain = Domain::normalized();
            let radius = 0.15;

            let c1 = SVector::from_fn(|_, _| 0.15);
            let c2 = SVector::from_fn(|_, _| 0.5);
            let sphere1 = Sphere::new(c1, radius, Some(Domain::normalized()));
            let sphere2 = Sphere::new(c2, radius, Some(Domain::normalized()));

            let mut classifier =
                SphereCluster::new(vec![sphere1, sphere2], Some(Domain::normalized()));

            let v = SVector::<f64, 10>::from_fn(|_, _| 1.0).normalize();
            let b = c1 - v * (radius - d * 0.9);

            assert!(
                classifier.classify(b).expect("Bug with invalid b.").class(),
                "b was not within mode"
            );

            let b2 =
                find_opposing_boundary(0.01, WithinMode(b), v, &domain, &mut classifier, 10, 10)
                    .expect("Unexpected error on sampling a constant location?");

            assert!(
                classifier
                    .classify(b2.into())
                    .expect("Unexpected out of bounds sample for opposing boundary sample?")
                    .class(),
                "Returned non-target (incorrect) sample?"
            );

            assert!(
                ((b2 - b).magnitude() - 2.0 * radius) <= 2.0 * d,
                "Resulting boundary point was not on opposite side of sphere?"
            );
        }
    }
}
