use crate::prelude::{Boundary, Classifier, Domain, Halfspace, Result, Sample};

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

        displacements.push(result.map(|new_hs| (new_hs.b - hs.b).norm()));
    }

    Ok((new_boundary, displacements))
}
