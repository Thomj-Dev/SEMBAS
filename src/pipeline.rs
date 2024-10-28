use crate::{
    prelude::{
        AdhererFactory, BoundaryPair, Classifier, Domain, Explorer, Halfspace, OutOfMode, Result,
        Sample, WithinMode,
    },
    search::{
        global_search::{DomainSampler, GlobalSearch, MonteCarloSampler, StandardGS},
        surfacing::binary_surface_search,
    },
};

/// TODO: Needs generics and to store the state of each stage.
pub enum Stage<E> {
    SearchingGlobally,
    Surfacing,
    Exploring(E),
}

/// Create a SEMBAS testing pipeline. Start by specifying the global search algorithm
/// and preferred setup method (auto/manual).
pub struct TestingPipeline<const N: usize, C: Classifier<N>> {
    classifier: C,
}

pub struct SurfacingStage<const N: usize, C, G>
where
    C: Classifier<N>,
    G: GlobalSearch<N, C>,
{
    classifier: C,
    gs: G,
}
pub struct AdhererStage<const N: usize, C, G, Su>
where
    C: Classifier<N>,
    G: GlobalSearch<N, C>,
    Su: Fn(&BoundaryPair<N>, &mut C) -> Vec<Halfspace<N>>,
{
    classifier: C,
    gs: G,
    surfacer: Su,
}
pub struct ExplorerStage<const N: usize, C, G, Su, F>
where
    C: Classifier<N>,
    G: GlobalSearch<N, C>,
    Su: Fn(&BoundaryPair<N>, &mut C) -> Vec<Halfspace<N>>,
    F: AdhererFactory<N>,
{
    classifier: C,
    gs: G,
    surfacer: Su,
    adherer_f: F,
}

pub struct Sembas<const N: usize, C, G, Su, F, E, Ef>
where
    C: Classifier<N>,
    G: GlobalSearch<N, C>,
    Su: Fn(&BoundaryPair<N>, &mut C) -> Vec<Halfspace<N>>,
    F: AdhererFactory<N>,
    E: Explorer<N, F>,
    Ef: Fn() -> E,
{
    classifier: C,
    gs: G,
    surfacer: Su,
    adherer_f: F,
    expl_f: Ef,
    stage: Stage<E>,
}

impl<const N: usize, C: Classifier<N>> TestingPipeline<N, C> {
    pub fn new(classifier: C) -> Self {
        Self { classifier }
    }

    pub fn monte_carlo(
        self,
        domain: Domain<N>,
        seed: u64,
    ) -> SurfacingStage<N, C, StandardGS<N, MonteCarloSampler<N>>> {
        SurfacingStage::new(self, StandardGS::new(MonteCarloSampler::new(domain, seed)))
    }
}

impl<const N: usize, C: Classifier<N>, G: GlobalSearch<N, C>> SurfacingStage<N, C, G> {
    fn new(prev: TestingPipeline<N, C>, gs: G) -> Self {
        Self {
            classifier: prev.classifier,
            gs,
        }
    }

    pub fn custom_surface_search<Su: Fn(&BoundaryPair<N>, &mut C) -> Vec<Halfspace<N>>>(
        self,
        surfacer: Su,
    ) -> AdhererStage<N, C, G, Su> {
        AdhererStage::new(self, surfacer)
    }

    pub fn binary_surface_search(
        self,
        max_err: f64,
        max_samples: u32,
    ) -> AdhererStage<N, C, G, impl Fn(&BoundaryPair<N>, &mut C) -> Vec<Halfspace<N>>> {
        AdhererStage::new(self, move |b_pair, classifier| {
            if let Ok(hs) = binary_surface_search::<N, C>(max_err, b_pair, max_samples, classifier)
            {
                vec![hs]
            } else {
                vec![]
            }
        })
    }
}

impl<const N: usize, C, G, Su> AdhererStage<N, C, G, Su>
where
    C: Classifier<N>,
    G: GlobalSearch<N, C>,
    Su: Fn(&BoundaryPair<N>, &mut C) -> Vec<Halfspace<N>>,
{
    fn new(prev: SurfacingStage<N, C, G>, surfacer: Su) -> Self {
        Self {
            classifier: prev.classifier,
            gs: prev.gs,
            surfacer,
        }
    }

    fn custom_adherer<F: AdhererFactory<N>>(adh_f: F) -> ExplorerStage<N, C, G, Su, F> {
        // ExplorerStage::
        todo!()
    }
}
impl<const N: usize, C, G, Su, F> ExplorerStage<N, C, G, Su, F>
where
    C: Classifier<N>,
    G: GlobalSearch<N, C>,
    Su: Fn(&BoundaryPair<N>, &mut C) -> Vec<Halfspace<N>>,
    F: AdhererFactory<N>,
{
    fn new(prev: AdhererStage<N, C, G, Su>, adherer_f: F) -> Self {
        Self {
            classifier: prev.classifier,
            gs: prev.gs,
            surfacer: prev.surfacer,
            adherer_f,
        }
    }
}

impl<const N: usize, C, G, Su, F, E, Ef> Sembas<N, C, G, Su, F, E, Ef>
where
    C: Classifier<N>,
    G: GlobalSearch<N, C>,
    Su: Fn(&BoundaryPair<N>, &mut C) -> Vec<Halfspace<N>>,
    F: AdhererFactory<N>,
    E: Explorer<N, F>,
    Ef: Fn() -> E,
{
    pub fn new(prev: ExplorerStage<N, C, G, Su, F>, expl_f: Ef) -> Self {
        Self {
            classifier: prev.classifier,
            gs: prev.gs,
            surfacer: prev.surfacer,
            adherer_f: prev.adherer_f,
            expl_f,
            stage: Stage::SearchingGlobally,
        }
    }

    pub fn step(&self) -> &Sample<N> {
        match self.stage {
            Stage::SearchingGlobally => {
                self.gs.step(&mut self.classifier);
                if let Some(bp) = self.gs.pop() {}
            }
            Stage::Surfacing => todo!(),
            Stage::Exploring(expl) => todo!(),
        }
    }
}
