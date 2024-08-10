use nalgebra::{vector, SVector};
use sembas::{
    adherers::const_adherer::ConstantAdhererFactory,
    explorer_core::Explorer,
    explorers::MeshExplorer,
    search::{
        global_search::{MonteCarloSearch, SearchFactory},
        surfacing::binary_surface_search,
    },
    structs::{BoundaryPair, Classifier, Domain, Sample, SamplingError},
};

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

fn main() {
    // Setup your classifier
    let mut classifier: Box<dyn Classifier<3>> = Box::new(Sphere {
        center: vector![0.5, 0.5, 0.5],
        radius: 0.25,
        domain: Domain::normalized(),
    });
    // Pick a (d)istance between samples:
    let d = 0.05;

    // Acquire an initial boundary pair using some global search solution
    println!("Finding initial boundary pair.");
    //      v--Stores intermediate results
    let mut history = vec![];

    let b_pair = find_initial_boundary_pair(&mut classifier, &mut history, 256).unwrap();
    println!("Pair found!");

    // Use the boundary pair to find the surface using binary surface search
    println!("Surfacing...");
    let root =
        binary_surface_search(d, &b_pair, 256, &mut classifier).expect("Couldn't find in time?");

    // Create an adherer factory that the explorer will use to adhere to the surface
    // * Extremely low delta angle and max rotation will result in Boundary Lost Exceptions (BLE)
    // * Fairly rare overall. Usually caused by an accumulated inaccuracy in the
    //   surface vector direction. Won't really happen naturally for a simple shape
    //   like a sphere.
    let delta_angle = 15.0f64.to_radians();
    let max_rotation = 180.0f64.to_radians(); // Will fail if it reach 180 deg.
    let adherer_f = ConstantAdhererFactory::new(delta_angle, Some(max_rotation));

    // Create your explorer
    let mut expl = MeshExplorer::new(d, root, d * 0.9, Box::new(adherer_f));

    let max_boundary_points = 500;
    let max_samples = 1000;
    let mut ble_count = 0;
    let mut oob_count = 0;

    println!("Beginning exploration process for a maximum of {max_boundary_points} boundary points or {max_samples} total samples.");
    for _ in 0..max_samples {
        // Take samples and handle results
        if let Err(e) = expl.step(&mut classifier) {
            match e {
                SamplingError::BoundaryLost => ble_count += 1,
                SamplingError::OutOfBounds => oob_count += 1,
                _ => (),
            }
        }

        if expl.boundary_count() >= max_boundary_points {
            break;
        }
    }

    // Do what you want with the results
    println!("Completed.");
    println!("Boundary points found: {}", expl.boundary_count());
    println!("BLE Count: {ble_count}");
    println!("OOB Count: {oob_count}");
}

fn find_initial_boundary_pair<const N: usize>(
    classifier: &mut Box<dyn Classifier<N>>,
    history: &mut Vec<Sample<N>>,
    max_samples: i32,
) -> Result<BoundaryPair<N>, SamplingError<N>> {
    let mut search = MonteCarloSearch::new(Domain::normalized(), 1);
    let mut take_sample = move || {
        let p = search.sample();
        let cls = classifier
            .classify(&p)
            .expect("Invalid sample. Bad global search domain?");
        Sample::from_class(p, cls)
    };

    let mut t0 = None;
    let mut x0 = None;
    let mut i = 0;

    while (t0.is_none() || x0.is_none()) && i < max_samples {
        let sample = take_sample();
        match sample {
            Sample::WithinMode(t) => {
                if t0.is_none() {
                    t0 = Some(t)
                }
            }
            Sample::OutOfMode(x) => {
                if x0.is_none() {
                    x0 = Some(x)
                }
            }
        }

        history.push(sample);

        i += 1;
    }

    if let (Some(t), Some(x)) = (t0, x0) {
        Ok(BoundaryPair::new(t, x))
    } else {
        Err(SamplingError::MaxSamplesExceeded)
    }
}