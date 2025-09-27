use std::{
    f64::consts::PI,
    fs::OpenOptions,
    io::{self, Write},
    path::Path,
};

use sembas::{
    api::SembasSession,
    boundary_tools::{estimation::approx_surface, reacquisition::reacquire_all_hybrid},
    prelude::{bs_adherer::BinarySearchAdhererFactory, *},
    search::{global_search::*, surfacing::binary_surface_search},
    structs::{
        messages::{MSG_PHASE_BOUNDARY_EXPL, MSG_PHASE_GLOBAL_SEARCH, MSG_PHASE_SURFACE_SEARCH},
        Classifier,
    },
};
use serde::{Deserialize, Serialize};

const NDIM: usize = 4;
// const JUMP_DIST: f64 = 0.075;
const JUMP_DIST: f64 = 0.025;
const ANGLE: f64 = 0.3;
const MSG_REACQUIRE: &str = "REACQ";

#[derive(Serialize, Deserialize)]
struct BoundaryData {
    boundary_points: Vec<Vec<f64>>,
    boundary_surface: Vec<Vec<f64>>,
}

//
fn main() {
    // let mut classifier = RemoteClassifier::<NDIM>::bind("127.0.0.1:2000".to_string()).unwrap();
    let mut classifier =
        SembasSession::<NDIM>::bind("127.0.0.1:2000".to_string(), MSG_PHASE_GLOBAL_SEARCH).unwrap();
    let mut i = 0;
    loop {
        let root = get_new_root(&mut classifier, i);
        // explore_and_expand(&mut classifier, root);
        explore(&mut classifier, root);
        i += 1;
    }
}

fn get_new_root<const N: usize>(classifier: &mut SembasSession<N>, seed: u64) -> Halfspace<N> {
    println!("Finding initial pair...");
    // classifier
    classifier.update_phase(MSG_PHASE_GLOBAL_SEARCH);
    let bp = find_initial_boundary_pair(classifier, 1000, seed).unwrap();

    // let roots: Vec<Halfspace<NDIM>> =
    //     find_chords(JUMP_DIST * 0.25, &bp, NDIM, &domain, classifier)
    //         .unwrap()
    //         .into_iter()
    //         .flat_map(|(a, b)| vec![a, b])
    //         .collect();

    // println!("Initial bp: {bp:?}");
    // println!("Roots: {roots:?}");
    println!("Establishing roots...");
    classifier.update_phase(MSG_PHASE_SURFACE_SEARCH);

    let root = binary_surface_search(JUMP_DIST, &bp, 100, classifier).unwrap();

    let adh_f = BinarySearchAdhererFactory::new(PI / 2.0, 3);

    return match approx_surface(JUMP_DIST, root, &adh_f, classifier) {
        Ok((hs, _, _)) => hs,
        Err(_) => root,
    };
}

/// Given a starting half space, it continuously explores the boundary. It will then
/// be trained on that boundary (mutating the boundary) at which point it will reac-
/// quire.
fn explore_and_expand<const N: usize>(classifier: &mut SembasSession<N>, root: Halfspace<N>) {
    let domain = Domain::normalized();
    let adh_f = ConstantAdhererFactory::new(ANGLE, None);

    let mut root = root;

    loop {
        println!("Starting boundary exploration");
        classifier.update_phase(MSG_PHASE_BOUNDARY_EXPL);
        let mut expl = MeshExplorer::new(JUMP_DIST, root, JUMP_DIST * 0.8, adh_f);

        loop {
            match expl.step(classifier) {
                Ok(None) => {
                    println!("Ran out of boundary to explore before experiment completion.");
                    return;
                }
                Err(e) => println!("Got error: {e:?}"),
                _ => {
                    if let Some(msg) = classifier.expect_msg().unwrap() {
                        println!("FUT updated, reacquiring boundary, {msg} (should be reacq)");
                        if msg != "REACQ" {
                            panic!("Did not receive request or REACQ message, but got '{msg}' instead?")
                        }
                        break;
                    }
                }
            }
        }

        println!("Saving boundary before reacquisition...");
        save_boundary(expl.boundary(), ".data/rl-boundary/pre_reacq.json").unwrap();

        println!("Reacquiring boundary");
        classifier.update_phase(MSG_REACQUIRE);
        let (boundary_update, distances) = reacquire_all_hybrid(
            classifier,
            expl.boundary(),
            &domain,
            JUMP_DIST * 5.0,
            JUMP_DIST / 2.0,
            1,
            20,
        )
        .unwrap();

        let movements: Vec<f64> = distances.into_iter().filter_map(|s| s).collect();
        let net_movement: f64 = movements.iter().sum();
        let min_movement = movements
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let max_movement = movements
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        let num_lost = expl.boundary().len() - movements.len();
        let total_bps = expl.boundary().len();

        println!("Lost {num_lost} out of {total_bps} b points");
        println!("net movement: {net_movement}");
        println!("min movement: {min_movement}");
        println!("max movement: {max_movement}");

        println!("Saving boundary after reacquisition...");
        let new_boundary: Vec<Halfspace<N>> = boundary_update.iter().filter_map(|x| *x).collect();
        save_boundary(&new_boundary, ".data/rl-boundary/post_reacq.json").unwrap();

        root = boundary_update
            .into_iter()
            .filter_map(|x| x)
            .next()
            .expect("Failed to reacquire the boundary");
    }
}
/// Given a starting half space, it continuously explores the boundary. It will then
/// be trained on that boundary (mutating the boundary) at which point it will reac-
/// quire.
fn explore<const N: usize>(classifier: &mut SembasSession<N>, root: Halfspace<N>) {
    let adh_f = ConstantAdhererFactory::new(ANGLE, None);

    println!("Starting boundary exploration");
    classifier.update_phase(MSG_PHASE_BOUNDARY_EXPL);
    let mut expl = MeshExplorer::new(JUMP_DIST, root, JUMP_DIST * 0.8, adh_f);

    loop {
        if let Some(msg) = classifier.expect_msg().unwrap() {
            println!("FUT updated, reacquiring boundary, {msg} (should be reacq)");
            if msg != "REACQ" {
                panic!("Did not receive request or REACQ message, but got '{msg}' instead?")
            }
            break;
        }
        match expl.step(classifier) {
            Ok(None) => {
                println!("Ran out of boundary to explore before experiment completion.");
                return;
            }
            Err(e) => println!("Got error: {e:?}"),
            _ => (),
        }
    }
}

fn find_initial_boundary_pair<const N: usize, C: Classifier<N>>(
    classifier: &mut C,
    max_samples: i32,
    seed: u64,
) -> Result<BoundaryPair<N>> {
    let mut search = MonteCarloSearch::new(Domain::normalized(), seed);
    let mut take_sample = move || {
        let p = search.sample();
        classifier
            .classify(p)
            .expect("Invalid sample. Bad global search domain?")
    };

    let mut t0 = None;
    let mut x0 = None;
    let mut i = 0;

    while (t0.is_none() || x0.is_none()) && i < max_samples {
        let sample = take_sample();
        match sample {
            Sample::WithinMode(t) => {
                if t0.is_none() {
                    println!("Found target");
                    t0 = Some(t)
                }
            }
            Sample::OutOfMode(x) => {
                if x0.is_none() {
                    println!("Found non-target");
                    x0 = Some(x)
                }
            }
        }

        i += 1;
    }

    if let (Some(t), Some(x)) = (t0, x0) {
        Ok(BoundaryPair::new(t, x))
    } else {
        Err(SamplingError::MaxSamplesExceeded)
    }
}

fn save_boundary<const N: usize>(boundary: &Boundary<N>, path: &str) -> io::Result<()> {
    let path = Path::new(path);
    if let Some(prefix) = path.parent() {
        std::fs::create_dir_all(prefix).unwrap();
    }
    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;

    let (boundary_points, boundary_surface): (Vec<Vec<f64>>, Vec<Vec<f64>>) = boundary
        .iter()
        .map(|hs| {
            (
                (*hs.b).iter().copied().collect(),
                hs.n.iter().copied().collect(),
            )
        })
        .unzip();

    f.write_all(
        serde_json::to_string_pretty(&BoundaryData {
            boundary_points,
            boundary_surface,
        })?
        .as_bytes(),
    )?;
    Ok(())
}
