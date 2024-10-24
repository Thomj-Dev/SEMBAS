use std::time::{Duration, Instant};

use nalgebra::SVector;
use sembas::{
    prelude::{ConstantAdhererFactory, Explorer, MeshExplorer},
    sps::Sphere,
    structs::{Halfspace, WithinMode},
};

#[test]
fn measure_sphere_expl_times() {
    const ITER_COUNT: u32 = 20;

    let mut sphere = Sphere::new(SVector::repeat(0.5), 0.5, None);

    let d = 0.01;
    let angle = 5.0f64.to_radians();
    let n0 = SVector::<f64, 10>::repeat(1.0).normalize();
    let b0 = WithinMode(n0 * (sphere.radius() - d * 0.25));
    let root = Halfspace { b: b0, n: n0 };

    let mut times = vec![];

    for _ in 0..ITER_COUNT {
        let adhf = ConstantAdhererFactory::new(angle, None);
        let mut expl = MeshExplorer::new(d, root, d * 0.9, adhf);

        let start_time = Instant::now();

        while let Ok(Some(_)) = expl.step(&mut sphere) {}

        times.push((Instant::now() - start_time).as_micros());
    }

    println!(
        "Mean time: {}us. Max time: {}us.",
        times.iter().sum::<u128>() as f64 / times.len() as f64,
        times.iter().max().unwrap()
    );
}
