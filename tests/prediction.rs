use std::{
    f64::consts::PI,
    time::{Duration, Instant},
};

use nalgebra::SVector;
use sembas::{
    boundary_tools::estimation::{
        approx_mc_volume, approx_mc_volume_intersection, approx_prediction, PredictionMode,
    },
    prelude::{ConstantAdhererFactory, Explorer, MeshExplorer},
    search::global_search::{MonteCarloSearch, SearchFactory},
    sps::{Cube, Sphere},
    structs::{Classifier, Domain, Halfspace, WithinMode},
};

const JUMP_DISTANCE: f64 = 0.1;
const MARGIN: f64 = JUMP_DISTANCE * 0.85;
const ADH_DELTA_ANGLE: f64 = 0.261799;
const ADH_MAX_ANGLE: f64 = std::f64::consts::PI;

fn setup_mesh_expl_sphere<const N: usize>(
    sphere: &Sphere<N>,
) -> MeshExplorer<N, ConstantAdhererFactory<N>> {
    let b = WithinMode(
        sphere.center() + SVector::from_fn(|i, _| if i == 0 { sphere.radius() } else { 0.0 }),
    );
    let mut n = SVector::zeros();
    n[0] = 1.0;
    let root = Halfspace { b, n };
    let adherer_f = ConstantAdhererFactory::new(ADH_DELTA_ANGLE, Some(ADH_MAX_ANGLE));

    MeshExplorer::new(JUMP_DISTANCE, root, MARGIN, adherer_f)
}

fn setup_mesh_expl_cube<const N: usize>(
    cube: &Cube<N>,
) -> MeshExplorer<N, ConstantAdhererFactory<N>> {
    let mut center = cube.shape().low() + cube.shape().dimensions() / 2.0;
    center[0] = cube.shape().high()[0];
    let b = WithinMode(center);

    let mut n = SVector::zeros();
    n[0] = 1.0;
    let root = Halfspace { b, n };
    let adherer_f = ConstantAdhererFactory::new(ADH_DELTA_ANGLE, Some(ADH_MAX_ANGLE));

    MeshExplorer::new(JUMP_DISTANCE, root, MARGIN, adherer_f)
}

#[test]
fn mode_pred() {
    const NDIM: usize = 3;
    let mut sphere = Sphere::<NDIM>::new(SVector::repeat(0.5), 0.25, None);
    // let area = sphere_surface_area(&sphere);
    let mut expl = setup_mesh_expl_sphere(&sphere);

    let timeout = Duration::from_secs(5);
    let start_time = Instant::now();

    loop {
        if let Ok(None) = expl.step(&mut sphere) {
            break;
        }
        if start_time.elapsed() > timeout {
            panic!("Test exceeded expected time to completion. Mesh explorer got stuck?");
        }
    }

    let mut mc =
        MonteCarloSearch::new(Domain::new(SVector::repeat(0.25), SVector::repeat(0.75)), 1);

    let points: Vec<SVector<f64, NDIM>> = (0..1000).map(|_| mc.sample()).collect();

    let truth: Vec<_> = points
        .iter()
        .map(|p| sphere.classify(*p).unwrap())
        .collect();
    let pred: Vec<_> = points
        .iter()
        .map(|p| approx_prediction(*p, expl.boundary(), expl.knn_index(), 1))
        .collect();

    let correct: Vec<_> = pred
        .iter()
        .zip(truth.iter())
        .filter(|(expected, test)| test.class() == expected.class())
        .collect();
    let accuracy = correct.len() as f64 / truth.len() as f64;

    assert!(
        accuracy > 0.75,
        "Excessive error in volume. accuracy: {accuracy}"
    );
}

#[test]
fn volume_mc() {
    const NDIM: usize = 3;
    let mut sphere = Sphere::<NDIM>::new(SVector::repeat(0.5), 0.25, None);
    let radius = sphere.radius();
    // let area = sphere_surface_area(&sphere);
    let mut expl = setup_mesh_expl_sphere(&sphere);

    let timeout = Duration::from_secs(5);
    let start_time = Instant::now();
    if start_time.elapsed() > timeout {
        panic!("Test exceeded expected time to completion. Mesh explorer got stuck?");
    }

    loop {
        if let Ok(None) = expl.step(&mut sphere) {
            break;
        }
        if start_time.elapsed() > timeout {
            panic!("Test exceeded expected time to completion. Mesh explorer got stuck?");
        }
    }

    let true_volume = 4.0 / 3.0 * PI * radius.powf(3.0);
    let est_vol = approx_mc_volume(
        PredictionMode::Intersection,
        &[(expl.boundary(), expl.knn_index())],
        1000,
        1,
        1,
    );

    let perc_err = (est_vol - true_volume).abs() / true_volume;
    assert!(
        perc_err < 0.2,
        "Excessive error in volume. err:{perc_err}, vol: {est_vol}, true vol: {true_volume}"
    );
}

#[test]
fn inscribed_sphere_has_no_distinct_volume() {
    const NDIM: usize = 3;

    let mut sphere = Sphere::<NDIM>::new(SVector::repeat(0.5), 0.25, None);
    let mut inner_sphere = Sphere::<NDIM>::new(SVector::repeat(0.5), 0.1, None);

    let mut expl1 = setup_mesh_expl_sphere(&sphere);
    let mut expl2 = setup_mesh_expl_sphere(&inner_sphere);
    let start_time = Instant::now();
    let timeout = Duration::from_secs(5);

    loop {
        if let Ok(None) = expl1.step(&mut sphere) {
            break;
        }
        if start_time.elapsed() > timeout {
            panic!("Test exceeded expected time to completion. Mesh explorer got stuck?");
        }
    }
    let start_time = Instant::now();
    loop {
        if let Ok(None) = expl2.step(&mut inner_sphere) {
            break;
        }
        if start_time.elapsed() > timeout {
            panic!("Test exceeded expected time to completion. Mesh explorer got stuck?");
        }
    }

    let (inter_vol, _, b_vol) = approx_mc_volume_intersection(
        &[(expl1.boundary(), expl1.knn_index())],
        &[(expl2.boundary(), expl2.knn_index())],
        100,
        1,
        1,
    );

    let b_prop = b_vol / (inter_vol + b_vol);

    assert!(
        b_prop < 0.05,
        "Estimated inner sphere distinct volume prediction was excessively high. Expected ~0 vol, got {b_vol}."
    );
}

#[test]
fn intersecting_vol_results_are_in_correct_order() {
    const NDIM: usize = 3;

    let mut sphere = Sphere::<NDIM>::new(
        SVector::from_fn(|i, _| if i == 0 { 0.0 } else { 0.5 }),
        0.5,
        None,
    );
    let mut cube = Cube::<NDIM>::new(Domain::normalized(), None);

    let mut expl1 = setup_mesh_expl_sphere(&sphere);
    let mut expl2 = setup_mesh_expl_cube(&cube);

    loop {
        if let Ok(None) = expl1.step(&mut sphere) {
            break;
        }
    }
    loop {
        if let Ok(None) = expl2.step(&mut cube) {
            break;
        }
    }

    let (inter_vol, sphere_v, _cube_v) = approx_mc_volume_intersection(
        &[(expl1.boundary(), expl1.knn_index())],
        &[(expl2.boundary(), expl2.knn_index())],
        1000,
        1,
        1,
    );

    assert!(
        (sphere_v - inter_vol).abs() / sphere_v < 1e-2,
        "Sphere - Cube != 1/2 Sphere. It is either due to estimation error or the return order is wrong. inter vol: {inter_vol}, 'sphere' vol: {sphere_v}, 'cube' vol: {_cube_v}"
    );
}
