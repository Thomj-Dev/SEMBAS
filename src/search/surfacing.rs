use crate::{
    prelude::Sample,
    structs::{BoundaryPair, Classifier, Halfspace, Result, SamplingError, WithinMode},
};

/// Finds the surface of an envelope, i.e. the initial halfspace for beginning
/// surface exploration by iteratively splitting the space in half until a desireable
/// distance from the boundary has been reached.
/// ## Arguments
/// * `max_err` The desired maximum distance from the boundary.
/// * `t0` A target sample
/// * `x0` A non-target sample
/// * `max_samples` The maximum number of samples before the failing the process.
pub fn binary_surface_search<const N: usize, C: Classifier<N>>(
    max_err: f64,
    b_pair: &BoundaryPair<N>,
    max_samples: u32,
    classifier: &mut C,
) -> Result<Halfspace<N>> {
    let mut p_t = b_pair.t().0;
    let mut p_x = b_pair.x().0;
    let mut s = p_x - p_t;
    let mut i = 0;

    while s.norm() > max_err && i < max_samples {
        s = (p_x - p_t) / 2.0;
        i += 1;

        match classifier.classify(p_t + s)? {
            Sample::WithinMode(_) => p_t += s,
            Sample::OutOfMode(_) => p_x -= s,
        }
    }

    if i >= max_samples && s.norm() > max_err {
        println!("Norm: {}", s.norm());
        return Err(SamplingError::MaxSamplesExceeded);
    }

    let n = s.normalize();

    Ok(Halfspace {
        b: WithinMode(p_t),
        n,
    })
}

#[cfg(all(test, feature = "sps"))]
mod test_surfacer {
    use nalgebra::{vector, SVector};

    use crate::{
        prelude::FunctionClassifier,
        sps::Sphere,
        structs::{BoundaryPair, Classifier, Domain, OutOfMode, SamplingError, WithinMode},
    };

    use super::binary_surface_search;

    const RADIUS: f64 = 0.25;
    const DIAMETER: f64 = 2.0 * RADIUS;

    fn setup_sphere<const N: usize>() -> Sphere<N> {
        let radius = 0.25;
        let center = SVector::from_fn(|_, _| 0.5);
        let domain = Domain::normalized();

        Sphere::new(center, radius, Some(domain))
    }

    #[test]
    fn finds_sphere_surface() {
        const D: usize = 10;
        let mut sphere = setup_sphere::<D>();
        let d = 0.05;

        // a little perturbed from the center
        let mut t = SVector::from_fn(|_, _| 0.5);
        t[D - 1] += 0.15;
        let t = WithinMode(t);
        let mut x = SVector::from_fn(|_, _| 0.15);
        x[0] += 0.3;
        let x = OutOfMode(x);

        let hs = binary_surface_search(d, &BoundaryPair::new(t, x), 100, &mut sphere)
            .expect("Failed to find boundary?");

        assert!(
            sphere.classify(*hs.b).unwrap().class(),
            "Halfspace outside of geometry?"
        );

        assert!(
            !sphere.classify(hs.b + hs.n * d).unwrap().class(),
            "Halfspace not on boundary?"
        );
    }

    #[test]
    fn finds_sphere_surface_within_maximum_samples() {
        const D: usize = 10;
        let mut sphere = setup_sphere::<D>();
        let d = 0.05;

        // a little perturbed from the center
        let mut t: SVector<f64, D> = SVector::from_fn(|_, _| 0.5);
        t[D - 1] += 0.15;
        let t = WithinMode(t);
        let mut x: SVector<f64, D> = SVector::from_fn(|_, _| 0.15);
        x[0] += 0.3;
        let x = OutOfMode(x);
        let max_samples = ((t - x).norm() / d).log2().ceil() as u32;

        let hs = binary_surface_search(d, &BoundaryPair::new(t, x), max_samples, &mut sphere)
            .expect("Failed to find boundary within the maximum number of samples ({max_samples})");

        assert!(
            sphere.classify(*hs.b).unwrap().class(),
            "Halfspace outside of geometry?"
        );
        assert!(
            !sphere.classify(hs.b + hs.n * d).unwrap().class(),
            "Halfspace not on boundary?"
        );
    }

    #[test]
    fn fails_finding_sphere_surface_below_expected_sample_count() {
        const D: usize = 10;
        let mut sphere = setup_sphere::<D>();
        let d = 0.05;

        // a little perturbed from the center
        let t = SVector::from_fn(|_, _| 0.5);
        let t = WithinMode(t);

        let mut x = SVector::from_fn(|_, _| 0.5);
        x[0] = 0.0;
        let x = OutOfMode(x);
        let minimum_sample_count = (RADIUS / d).log2().ceil() as u32;

        let err: SamplingError = binary_surface_search(
            d,
            &BoundaryPair::new(t, x),
            minimum_sample_count - 1,
            &mut sphere,
        )
        .expect_err(
            format!(
                "Found boundary despite not having enough samples ({})?",
                minimum_sample_count - 1
            )
            .as_str(),
        );
        assert_eq!(
            err,
            SamplingError::MaxSamplesExceeded,
            "Unexpected error type? Got: {err:?}"
        )
    }

    #[test]
    fn finds_boundary_within_max_dist() {
        let max_err = 0.05;
        let domain = Domain::<3>::normalized();
        let mut classifier = FunctionClassifier::new(|x| {
            if domain.contains(&x) {
                Ok(x[0] < 0.75)
            } else {
                Err(SamplingError::OutOfBounds)
            }
        });

        let result = binary_surface_search(
            max_err,
            &BoundaryPair::new(
                WithinMode(vector![0.5, 0.5, 0.5]),
                OutOfMode(vector![1.0, 0.5, 0.0]),
            ),
            20,
            &mut classifier,
        )
        .expect("Got error when expecting results from BSS");

        let dist = (result.b[0] - 0.75).abs();
        assert!(
            dist <= max_err,
            "Got a distance from boundary greater than max dist: {dist} > {max_err}"
        );
    }
}
