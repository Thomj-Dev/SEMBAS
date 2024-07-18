use std::time::{Duration, Instant};

use nalgebra::SVector;
use sembas::{
    adherers::const_adherer::ConstantAdhererFactory,
    explorer_core::Explorer,
    explorers::MeshExplorer,
    structs::{Classifier, Domain, Halfspace, SamplingError, WithinMode},
};

const D: usize = 10;
const JUMP_DISTANCE: f64 = 0.2;
const ADH_DELTA_ANGLE: f64 = 0.261799;
const ADH_MAX_ANGLE: f64 = std::f64::consts::PI;

// const ATOL: f64 = 1e-10;

struct Sphere<const N: usize> {
    pub radius: f64,
    pub center: SVector<f64, N>,
    pub domain: Domain<N>,
}

impl<const N: usize> Classifier<N> for Sphere<N> {
    fn classify(&mut self, p: &SVector<f64, N>) -> Result<bool, SamplingError<N>> {
        if !self.domain.contains(p) {
            return Err(SamplingError::OutOfBounds);
        }

        Ok((p - self.center).norm() <= self.radius)
    }
}

fn setup_mesh_expl<const N: usize>(sphere: &Sphere<N>) -> MeshExplorer<N> {
    let b = WithinMode(SVector::from_fn(|i, _| {
        if i == 0 {
            0.49 + sphere.radius
        } else {
            0.5
        }
    }));
    let mut n = SVector::zeros();
    n[0] = 1.0;
    let root = Halfspace { b, n };
    let adherer_f = ConstantAdhererFactory::new(ADH_DELTA_ANGLE, Some(ADH_MAX_ANGLE));

    MeshExplorer::new(
        JUMP_DISTANCE,
        root,
        JUMP_DISTANCE * 0.85,
        Box::new(adherer_f),
    )
}

fn setup_sphere<const N: usize>() -> Sphere<N> {
    let radius = 0.25;
    let center = SVector::from_fn(|_, _| 0.5);
    let domain = Domain::normalized();

    Sphere {
        radius,
        center,
        domain,
    }
}

fn sphere_to_classifier<const N: usize>(sphere: Sphere<N>) -> Box<dyn Classifier<N>> {
    Box::new(sphere)
}

fn average_vectors<const N: usize>(vectors: &Vec<SVector<f64, N>>) -> Option<SVector<f64, N>> {
    if vectors.is_empty() {
        return None; // Return None if the input vector is empty
    }

    // Initialize a vector to accumulate sums of components
    let mut sum_vector = SVector::<f64, N>::zeros();

    // Sum all vectors component-wise
    for vector in vectors {
        sum_vector += vector;
    }

    // Calculate the average vector
    let num_vectors = vectors.len() as f64;
    let average_vector = sum_vector / num_vectors;

    Some(average_vector)
}

// fn sphere_surface_area(sphere: &Sphere<10>) -> f64 {
//     (2.0 * std::f64::consts::PI.powf(5.0)) / (120.0) * sphere.radius.powf(5.0)
// }

// fn approx(a: f64, b: f64, atol: f64) -> bool {
//     (a - b).abs() < atol
// }

#[test]
fn fully_explores_sphere() {
    let sphere = setup_sphere::<D>();
    let center = sphere.center;
    let radius = sphere.radius;
    // let area = sphere_surface_area(&sphere);
    let mut expl = setup_mesh_expl(&sphere);
    let mut classifier = sphere_to_classifier(sphere);

    let timeout = Duration::from_secs(5);
    let start_time = Instant::now();

    while let Ok(Some(_)) = expl.step(&mut classifier) {
        if start_time.elapsed() > timeout {
            panic!("Test exceeded expected time to completion. Mesh explorer got stuck?");
        }
    }

    // In order to know that we explored the sphere, we need to know it covered the
    // full shape. To do this, we can find the average position and make sure it was
    // close to the center.
    let boundary_points = expl.boundary().iter().map(|x| *x.b).collect();
    let center_of_mass = average_vectors(&boundary_points).expect("Empty boundary?");

    let avg_dist_from_center = (center_of_mass - center).norm();
    assert!(
        avg_dist_from_center < radius / 2.0,
        "Avg distance from center, {avg_dist_from_center}, was not less than 1/2 radius?"
    );

    // let ideal_points_per_area = 1.0 / JUMP_DISTANCE.powf((D - 1) as f64);
    // let measured_points_per_area = expl.boundary_count() as f64 / area;
    // assert!(
    //     ((measured_points_per_area - ideal_points_per_area) / ideal_points_per_area).abs() < 0.1,
    //     "Exceeded 10% error from ideal surface coverage? Ideal p/A: {ideal_points_per_area}, Actual p/A: {measured_points_per_area}"
    // );
    // println!("{measured_points_per_area} - {ideal_points_per_area} = {}", measured_points_per_area - ideal_points_per_area);
}