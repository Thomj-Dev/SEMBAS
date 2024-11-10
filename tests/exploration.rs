#![cfg(feature = "sps")]

use std::{
    f64::consts::PI,
    time::{Duration, Instant},
};

use nalgebra::{vector, SVector};
use petgraph::graph::NodeIndex;
use sembas::{
    adherers::const_adherer::ConstantAdhererFactory,
    boundary_tools::estimation::approx_prediction,
    explorer_core::Explorer,
    explorers::MeshExplorer,
    sps::Sphere,
    structs::{
        backprop::Backpropagation, Classifier, Domain, Halfspace, Result, Sample, SamplingError,
        WithinMode,
    },
};

const D: usize = 10;
const JUMP_DISTANCE: f64 = 0.2;
const MARGIN: f64 = JUMP_DISTANCE * 0.85;
const ADH_DELTA_ANGLE: f64 = 0.261799;
const ADH_MAX_ANGLE: f64 = std::f64::consts::PI;

fn setup_mesh_expl<const N: usize>(
    sphere: &Sphere<N>,
) -> MeshExplorer<N, ConstantAdhererFactory<N>> {
    let b = WithinMode(SVector::from_fn(|i, _| {
        if i == 0 {
            0.49 + sphere.radius()
        } else {
            0.5
        }
    }));
    let mut n = SVector::zeros();
    n[0] = 1.0;
    let root = Halfspace { b, n };
    let adherer_f = ConstantAdhererFactory::new(ADH_DELTA_ANGLE, Some(ADH_MAX_ANGLE));

    MeshExplorer::new(JUMP_DISTANCE, root, MARGIN, adherer_f)
}

fn setup_sphere<const N: usize>() -> Sphere<N> {
    let radius = 0.25;
    let center = SVector::from_fn(|_, _| 0.5);
    let domain = Domain::normalized();

    Sphere::new(center, radius, Some(domain))
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

#[test]
fn fully_explores_sphere() {
    let mut sphere = setup_sphere::<D>();
    let center = *sphere.center();
    let radius = sphere.radius();
    // let area = sphere_surface_area(&sphere);
    let mut expl = setup_mesh_expl(&sphere);

    let timeout = Duration::from_secs(5);
    let start_time = Instant::now();
    let mut i = 0;

    while let Ok(Some(_)) = expl.step(&mut sphere) {
        if start_time.elapsed() > timeout {
            panic!("Test exceeded expected time to completion. Mesh explorer got stuck?");
        }

        i += 1;
    }

    let osv_err: f64 = expl
        .boundary()
        .iter()
        .map(|hs| (hs.b - center).angle(&hs.n) / PI)
        .sum();

    let osv_err = osv_err / expl.boundary_count() as f64;

    println!(
        "Effiency: {}, osv err: {osv_err}",
        expl.boundary_count() as f64 / (i - expl.boundary_count()) as f64
    );

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
}

#[cfg(feature = "io")]
#[test]
fn saves_and_loads_results_correctly() {
    use sembas::structs::report::ExplorationStatus;

    let mut sphere = setup_sphere::<D>();
    // let area = sphere_surface_area(&sphere);
    let mut expl = setup_mesh_expl(&sphere);

    let timeout = Duration::from_secs(5);
    let start_time = Instant::now();

    while let Ok(Some(_)) = expl.step(&mut sphere) {
        if start_time.elapsed() > timeout {
            panic!("Test exceeded expected time to completion. Mesh explorer got stuck?");
        }
    }

    const DIR: &str = "tmp-testdata";
    const FILE_NAME: &str = "tmp.json";
    let path: String = format!("{DIR}/{FILE_NAME}");

    std::fs::create_dir_all(DIR).unwrap();

    let status = expl.describe();
    status.save(&path).unwrap();

    let loaded_status: ExplorationStatus<10, ConstantAdhererFactory<10>> =
        ExplorationStatus::load(&path).unwrap();

    assert!(
        loaded_status
            .boundary_points()
            .iter()
            .zip(expl.boundary().iter())
            .all(|(p, hs)| { SVector::<f64, 10>::from_column_slice(p) == *hs.b }),
        "One or more boundary points were incorrectly stored in json?"
    );

    assert!(
        loaded_status
            .boundary_surface()
            .iter()
            .zip(expl.boundary().iter())
            .all(|(n, hs)| { SVector::<f64, 10>::from_column_slice(n) == hs.n }),
        "One or more boundary points were incorrectly stored in json?"
    );

    std::fs::remove_file(&path).unwrap();
    std::fs::remove_dir_all(DIR).unwrap();
}

#[test]
fn backprop_fully_explores_sphere() {
    let mut sphere = setup_sphere::<D>();
    let center = *sphere.center();
    let radius = sphere.radius();
    let mut expl = setup_mesh_expl(&sphere);

    let timeout = Duration::from_secs(5);
    let start_time = Instant::now();
    let mut i = 0;
    let mut j = 0;

    while let Ok(Some(_)) = expl.step(&mut sphere) {
        if start_time.elapsed() > timeout {
            panic!("Test exceeded expected time to completion. Mesh explorer got stuck?");
        }

        if j != expl.boundary_count() {
            j = expl.boundary_count();
            expl.backprop(NodeIndex::new(j - 1), JUMP_DISTANCE * 1.5);
        }

        i += 1;
    }

    let osv_err: f64 = expl
        .boundary()
        .iter()
        .map(|hs| (hs.b - center).angle(&hs.n) / PI)
        .sum();

    let osv_err = osv_err / expl.boundary_count() as f64;

    println!(
        "Effiency: {}, osv err: {osv_err}",
        expl.boundary_count() as f64 / (i - expl.boundary_count()) as f64
    );

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
}

#[test]
fn oob_err_prunes_exploration_branch() {
    struct TestClassifier<const N: usize> {
        i: usize,
    }
    impl<const N: usize> Classifier<N> for TestClassifier<N> {
        fn classify(&mut self, p: SVector<f64, N>) -> Result<Sample<N>> {
            if self.i > 2 {
                Err(SamplingError::OutOfBounds)
            } else {
                self.i += 1;
                Ok(Sample::from_class(p, true))
            }
        }
    }
    let mut classifier = TestClassifier::<10> { i: 0 };

    let b = WithinMode(SVector::from_fn(|_, _| 0.5));
    let mut n = SVector::zeros();
    n[0] = 1.0;
    let root = Halfspace { b, n };
    let adherer_f = ConstantAdhererFactory::new(ADH_DELTA_ANGLE, Some(ADH_MAX_ANGLE));

    let mut expl = MeshExplorer::new(JUMP_DISTANCE, root, JUMP_DISTANCE * 0.85, adherer_f);

    let mut is_exploring = true;
    let start = Instant::now();
    while is_exploring {
        if let Ok(None) = expl.step(&mut classifier) {
            is_exploring = false;
        }

        if start.elapsed() > Duration::from_secs(5) {
            panic!("Explorer hung due to out of bounds exceptions!");
        }
    }
}

#[test]
fn ble_err_prunes_exploration_branch() {
    struct TestClassifier<const N: usize> {}
    impl<const N: usize> Classifier<N> for TestClassifier<N> {
        fn classify(&mut self, p: SVector<f64, N>) -> Result<Sample<N>> {
            Ok(Sample::from_class(p, true))
        }
    }
    let mut classifier = TestClassifier::<10> {};

    let b = WithinMode(SVector::from_fn(|_, _| 0.5));
    let mut n = SVector::zeros();
    n[0] = 1.0;
    let root = Halfspace { b, n };
    let adherer_f = ConstantAdhererFactory::new(ADH_DELTA_ANGLE, Some(ADH_MAX_ANGLE));

    let mut expl = MeshExplorer::new(JUMP_DISTANCE, root, JUMP_DISTANCE * 0.85, adherer_f);

    let mut is_exploring = true;
    let start = Instant::now();
    while is_exploring {
        if let Ok(None) = expl.step(&mut classifier) {
            is_exploring = false
        }

        if start.elapsed() > Duration::from_secs(5) {
            panic!("Explorer hung due to boundary lost errors!")
        }
    }
}

pub fn svector_to_array<const N: usize>(v: SVector<f64, N>) -> [f64; N] {
    v.as_slice()
        .try_into()
        .expect("Failed to convert slice to array.")
}

#[test]
fn prediction_tests() {
    let mut sphere = setup_sphere::<3>();
    let radius = sphere.radius();
    // let area = sphere_surface_area(&sphere);
    let mut expl = setup_mesh_expl(&sphere);

    let timeout = Duration::from_secs(5);
    let start_time = Instant::now();

    while let Ok(Some(_)) = expl.step(&mut sphere) {
        if start_time.elapsed() > timeout {
            panic!("Test exceeded expected time to completion. Mesh explorer got stuck?");
        }
    }

    let within_mode_points = [
        vector![0.5, 0.5, 0.5],
        vector![0.5 + radius - JUMP_DISTANCE, 0.5, 0.5],
    ];
    let out_of_mode_points = [
        vector![0.0, 0.0, 0.0],
        vector![0.5 + radius + JUMP_DISTANCE, 0.5, 0.5],
    ];

    for t in within_mode_points.iter() {
        if let Sample::OutOfMode(_) = approx_prediction(*t, expl.boundary(), expl.knn_index(), 1) {
            panic!("False Negative for within mode point {t:?}.")
        }
    }
    for x in out_of_mode_points.iter() {
        if let Sample::WithinMode(_) = approx_prediction(*x, expl.boundary(), expl.knn_index(), 1) {
            panic!("False Negative for within mode point {x:?}.")
        }
    }
}
