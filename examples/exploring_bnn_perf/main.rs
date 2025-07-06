use std::{
    fs::OpenOptions,
    io::{self, Write},
    path::Path,
};

use sembas::{
    api::RemoteClassifier,
    boundary_tools::{
        bulk_insert_rtree,
        estimation::{approx_mc_volume_intersection, approx_surface},
        falls_on_boundary, get_rtree_from_boundary,
    },
    metrics::find_chords,
    prelude::*,
    search::global_search::*,
    structs::{Classifier, Halfspace},
};
use serde::{Deserialize, Serialize};

const NDIM: usize = 2;
const JUMP_DIST: f64 = 0.01;
const ANGLE: f64 = 0.0873; // 5 deg

#[derive(Serialize, Deserialize)]
struct BoundaryData {
    boundary_points: Vec<Vec<f64>>,
    boundary_surface: Vec<Vec<f64>>,
}

/// In this example, we will look at how we can use SEMBAS to identify complementary
/// neural networks for constructing an ensemble from a
/// Bayesian Neural Network (BNN).
/// * The BNN is defined in `fut.py` using pytorch. We train the BNN using a limited
///   number of training examples, and sample from the BNN's distribution several
///   concrete neural networks (NN).
/// * SEMBAS is used to explores the region of validity for these NNs to identify
///   where they perform well.
/// * We then use these boundaries to determine which NNs should be used to compose
///   our ensemble by preferring unique regions of validity over redundant ones (i.e.
///   minimizing their overlap).
/// * Finally, we use these boundaries from the perspective of FUT to place greater
///   confidence on inputs into the ensemble that are within their region of validity
///   over those that are outside of the region of validity, to further increase
///   model performance.
fn main() {
    const NUM_NETWORKS: u32 = 1000;

    let mut boundaries: Vec<Vec<Halfspace<NDIM>>> = vec![];
    let mut btrees = vec![];
    let mut skiplist = vec![];

    for i in 0..NUM_NETWORKS {
        if let Ok((boundary, btree)) = explore_network() {
            let envelopes: Vec<(&[Halfspace<NDIM>], &BoundaryRTree<NDIM>)> = boundaries
                .iter()
                .zip(btrees.iter())
                .map(|(b, bt)| (b.as_slice(), bt))
                .collect();

            save_boundary(
                &boundary,
                format!(".data/boundaries/boundary_{i}.json").as_str(),
            )
            .unwrap();

            if evaluate(&boundary, &btree, envelopes.as_slice()) {
                boundaries.push(boundary);
                btrees.push(btree);
            } else {
                skiplist.push(i);
            }
        } else {
            skiplist.push(i);
        }
    }

    println!("Skiplist: {skiplist:?}");
}

fn evaluate<const N: usize>(
    boundary: &Boundary<N>,
    btree: &BoundaryRTree<N>,
    others: &[(&Boundary<N>, &BoundaryRTree<N>)],
) -> bool {
    if others.is_empty() {
        true
    } else {
        let (inter_vol, b_vol, _other_vol) =
            approx_mc_volume_intersection(&[(boundary, btree)], others, 100, 1, 1);

        inter_vol / (inter_vol + b_vol) < 0.2
    }
}

fn save_boundary<const N: usize>(boundary: &Boundary<N>, path: &str) -> io::Result<()> {
    let path = Path::new(path);
    if let Some(prefix) = path.parent() {
        std::fs::create_dir_all(prefix)?;
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

fn explore_network() -> Result<(Vec<Halfspace<2>>, BoundaryRTree<2>)> {
    // Setting up connection. Note that the SEMBAS server must run first, prior
    // to fut.py client
    let domain = Domain::normalized();
    let mut classifier = RemoteClassifier::<NDIM>::bind("127.0.0.1:2000".to_string()).unwrap();

    println!("Finding initial pair...");
    let bp = find_initial_boundary_pair(&mut classifier, 1000)?;
    println!("Establishing roots...");
    let roots: Vec<Halfspace<NDIM>> =
        find_chords(JUMP_DIST * 0.25, &bp, NDIM, &domain, &mut classifier)
            .unwrap()
            .into_iter()
            .flat_map(|(a, b)| vec![a, b])
            .collect();

    let adh_f = ConstantAdhererFactory::new(ANGLE, None);

    let mut full_boundary = vec![];
    let mut full_btree: Option<BoundaryRTree<NDIM>> = None;

    for root in roots {
        // improve surface approximation
        println!("Improving initial node surface approx...");
        let hs = match approx_surface(JUMP_DIST, root, &adh_f, &mut classifier) {
            Ok((hs, _, _)) => hs,
            Err(_) => root,
        };

        println!("Checking if already explored...");
        // skip if already explored
        if let Some(btree) = &full_btree {
            if falls_on_boundary(JUMP_DIST, &hs, &full_boundary, btree) {
                continue;
            }
        }

        println!("Not in boundary, proceeding with exploration...");
        let mut expl = MeshExplorer::new(JUMP_DIST, hs, JUMP_DIST * 0.8, adh_f);

        explore_boundary(&mut expl, &mut classifier);

        println!("Exploration complete. Adding to solution...");
        // Append to full boundary
        let mut boundary: Vec<Halfspace<2>> = expl.boundary_owned();

        if let Some(full_index) = full_btree.as_mut() {
            bulk_insert_rtree(full_index, &boundary);
        } else {
            full_btree = Some(get_rtree_from_boundary(&boundary));
        }

        full_boundary.append(&mut boundary);
    }

    if let Some(full_btree) = full_btree {
        Ok((full_boundary, full_btree))
    } else {
        Err(SamplingError::BoundaryLost)
    }
}

fn explore_boundary<const N: usize, F: AdhererFactory<N>, E: Explorer<N, F>, C: Classifier<N>>(
    explorer: &mut E,
    classifier: &mut C,
) {
    loop {
        if let Ok(None) = explorer.step(classifier) {
            break;
        }
    }
}

fn find_initial_boundary_pair<const N: usize, C: Classifier<N>>(
    classifier: &mut C,
    max_samples: i32,
) -> Result<BoundaryPair<N>> {
    let mut search = MonteCarloSearch::new(Domain::normalized(), 1);
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
                    t0 = Some(t)
                }
            }
            Sample::OutOfMode(x) => {
                if x0.is_none() {
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
