use crate::{
    prelude::{
        Boundary, BoundaryPair, Classifier, Domain, Halfspace, OutOfMode, Result, Sample,
        WithinMode,
    },
    search::{binary_search_between, surfacing::binary_surface_search, SearchMode},
};

/// Acquires the EXACT boundary for a given outdated halfspace.
///
/// This is used when the FUT has undergone some transformation, leading to
/// boundary data being invalidated.
///
/// ### Return
/// - Ok(Some(hs)) : The halfspace that was successfully reacquired
/// - Ok(None) : Failed to find the boundary.
/// - Err(SamplingError) : Classifier induced error, generally unexpected unless @domain is
///     incorret.
fn reacquire_hs_incremental<const N: usize, C>(
    classifier: &mut C,
    hs: &Halfspace<N>,
    domain: &Domain<N>,
    max_err: f64,
    max_samples: Option<u32>,
) -> Result<Option<Halfspace<N>>>
where
    C: Classifier<N>,
{
    let mut prev_sample = classifier.classify(*hs.b)?;
    let init_cls = prev_sample.class();

    let s = (if init_cls { 1.0 } else { -1.0 }) * max_err * hs.n;
    if !domain.contains(&(prev_sample.into_inner() + s)) {
        return Ok(None);
    }

    let mut sample = classifier.classify(prev_sample.into_inner() + s)?;

    let mut i = 0;

    while max_samples.is_none_or(|m| i < m)
        && domain.contains(&(sample.into_inner() + s))
        && sample.class() == init_cls
    {
        prev_sample = sample;
        sample = classifier.classify(sample.into_inner() + s)?;
        i += 1;
    }

    let boundary_exists = sample.class() != init_cls;

    match (boundary_exists, domain.contains(&sample), sample) {
        (true, true, Sample::WithinMode(b)) => Ok(Some(Halfspace { b: b, n: hs.n })),
        (true, _, _) => {
            if let Sample::WithinMode(b) = prev_sample {
                Ok(Some(Halfspace { b, n: hs.n }))
            } else {
                Ok(None)
            }
        }
        _ => Ok(None),
    }
}

/// Without jump distance
/// 1. Find edge of domain
/// 2. binary search between
///
/// With jump distance
/// 1. Create next point by p + s, where |s| = jump distance
/// 2. If p' falls outside of domain, map p' to domain edge
/// 3. If p' results in a boundary pair, binary search between p and p'
///    otherwise, repeat from (1)
pub fn reacquire_hs_bs<const N: usize, C>(
    classifier: &mut C,
    hs: &Halfspace<N>,
    domain: &Domain<N>,
    max_err: f64,
    max_samples: u32,
) -> Result<Option<Halfspace<N>>>
where
    C: Classifier<N>,
{
    let b_sample = classifier.classify(*hs.b)?;
    let new_b_direction = if b_sample.class() { hs.n } else { -hs.n };

    let edge = domain.clip_vector(&(
        hs.b + new_b_direction * domain.distance_to_edge(&hs.b, &new_b_direction)
            .expect("Invalid out of bounds HS? Reacquiring a boundary assumes you know where the original boundary was, which cannot exist outside of domain")
    ));
    let edge_sample = classifier.classify(edge)?;

    Ok(match (b_sample, edge_sample) {
        (Sample::WithinMode(t0), Sample::WithinMode(t1)) => {
            if let Some(nt0) =
                binary_search_between(SearchMode::Full, false, max_samples, *t0, *t1, classifier)
            {
                Some(binary_surface_search(
                    max_err,
                    &BoundaryPair::new(t0, OutOfMode(nt0)),
                    max_samples,
                    classifier,
                )?)
            } else {
                None
            }
        }
        (Sample::OutOfMode(nt0), Sample::OutOfMode(nt1)) => {
            if let Some(t0) =
                binary_search_between(SearchMode::Full, true, max_samples, *nt0, *nt1, classifier)
            {
                Some(binary_surface_search(
                    max_err,
                    &BoundaryPair::new(WithinMode(t0), nt0),
                    max_samples,
                    classifier,
                )?)
            } else {
                None
            }
        }
        (Sample::WithinMode(t0), Sample::OutOfMode(nt0))
        | (Sample::OutOfMode(nt0), Sample::WithinMode(t0)) => Some(binary_surface_search(
            max_err,
            &BoundaryPair::new(t0, nt0),
            max_samples,
            classifier,
        )?),
    })
}

/// Without jump distance
/// 1. Find edge of domain
/// 2. binary search between
///
/// With jump distance
/// 1. Create next point by p + s, where |s| = jump distance
/// 2. If p' falls outside of domain, map p' to domain edge
/// 3. If p' results in a boundary pair, binary search between p and p'
///    otherwise, repeat from (1)
pub fn reacquire_hs_hybrid<const N: usize, C>(
    classifier: &mut C,
    hs: &Halfspace<N>,
    domain: &Domain<N>,
    jump_dist: f64,
    max_err: f64,
    max_samples: u32,
) -> Result<Option<Halfspace<N>>>
where
    C: Classifier<N>,
{
    let b_sample = classifier.classify(*hs.b)?;
    let new_b_direction = if b_sample.class() { hs.n } else { -hs.n };
    let s = jump_dist * new_b_direction;

    let mut prev_sample = b_sample;
    let mut next_p = *prev_sample + s;

    let bp = loop {
        if domain.contains(&next_p) {
            break None;
        }
        let next_sample = classifier.classify(next_p)?;

        if next_sample.class() != b_sample.class() {
            // pair found on next_p
            break Some(
                BoundaryPair::from_samples(prev_sample, next_sample)
                    .expect("Next.class != b.class, yet not bound pair?"),
            );
        }

        if let Some(p) = binary_search_between(
            SearchMode::Full,
            !b_sample.class(),
            max_samples,
            *prev_sample,
            next_p,
            classifier,
        ) {
            break Some(
                BoundaryPair::from_samples(
                    prev_sample,
                    Sample::from_class(p, !prev_sample.class()),
                )
                .expect("Next.class != b.class, yet not bound pair?"),
            );
        }

        prev_sample = next_sample;
        next_p = *prev_sample + s;
    };

    let bp = if bp.is_none() && !domain.contains(&next_p) {
        // final check by getting edge
        let edge = domain.clip_vector(&next_p);
        if let Some(p) = binary_search_between(
            SearchMode::Full,
            !b_sample.class(),
            max_samples,
            *prev_sample,
            edge,
            classifier,
        ) {
            Some(
                BoundaryPair::from_samples(prev_sample, Sample::from_class(p, !b_sample.class()))
                    .expect("Next.class != b.class, yet not bound pair?"),
            )
        } else {
            None
        }
    } else {
        bp
    };

    if let Some(bp) = bp {
        Ok(Some(binary_surface_search(
            max_err,
            &bp,
            max_samples,
            classifier,
        )?))
    } else {
        Ok(None)
    }
}

/// Attempts to reacquire the EXACT boundary after the FUT has changed in some way.
///
/// "Incremental" means that fixed-sized jump distances are used across all boundary
/// points within @boundary.
///
/// ### Return
/// Ok
/// - new_boundary : The resultant boundary
/// - displacements : corresponding displacements for each halfspace in the @boundary
/// ERR : Classifier induced error, generally unexpected unless @domain is
///     incorret.
pub fn reacquire_all_incremental<const N: usize, C>(
    classifier: &mut C,
    boundary: &Boundary<N>,
    domain: &Domain<N>,
    max_err: f64,
    samples_per_hs: Option<u32>,
) -> Result<(Vec<Option<Halfspace<N>>>, Vec<Option<f64>>)>
where
    C: Classifier<N>,
{
    let mut new_boundary = vec![];
    let mut displacements = vec![];

    for hs in boundary {
        let result = reacquire_hs_incremental(classifier, hs, domain, max_err, samples_per_hs)?;
        new_boundary.push(result);

        displacements.push(result.map(|new_hs| {
            let s = new_hs.b - hs.b;
            if s.dot(&new_hs.n) > 0.0 {
                s.norm()
            } else {
                -s.norm()
            }
        }));
    }

    Ok((new_boundary, displacements))
}
