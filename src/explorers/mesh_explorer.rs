use std::{any::type_name, collections::HashMap};

use crate::{
    adherer_core::{Adherer, AdhererFactory, AdhererState},
    explorer_core::Explorer,
    extensions::Queue,
    prelude::{report::ExplorationStatus, KnnNode, NodeID},
    structs::{backprop::Backpropagation, Classifier, Halfspace, Result, Sample, Span},
    utils::{array_distance, svector_to_array},
};
use nalgebra::{self, Const, OMatrix, SVector};
use petgraph::{graph::NodeIndex, visit::EdgeRef, Direction::Incoming, Graph};
use rstar::{primitives::GeomWithData, RTree};

pub type Path<const N: usize> = (NodeID, SVector<f64, N>);

/// Explores a surface uniformly by using a grid-search approach.
pub struct MeshExplorer<const N: usize, F: AdhererFactory<N>> {
    d: f64,
    boundary: Vec<Halfspace<N>>,
    margin: f64,
    basis_vectors: OMatrix<f64, Const<N>, Const<N>>,
    path_queue: Vec<Path<N>>,
    current_parent: NodeID,
    tree: Graph<Halfspace<N>, ()>,
    knn_index: RTree<KnnNode<N>>,
    adherer: Option<F::TargetAdherer>,
    adherer_f: F,
}

impl<const N: usize, F: AdhererFactory<N>> MeshExplorer<N, F> {
    /// Creates a MeshExplorer instance.
    /// ## Arguments
    /// * d: The jump distance between boundary points. Describes how far apart the
    ///   samples are taken.
    /// * root: The initial boundary halfspace to begin exploration from.
    /// * margin: 0 < margin < d, The minimum distance between a sample and a known
    ///   halfspace before a path along a cardinal direction is rejected.
    ///   Values that are closer to d improve efficiency at the cost of coverage.
    ///   If uncertain of what value to use, try 0.9 * d to start.
    pub fn new(d: f64, root: Halfspace<N>, margin: f64, adherer_f: F) -> Self {
        let boundary = vec![root];
        let basis_vectors = OMatrix::<f64, Const<N>, Const<N>>::identity();
        let path_queue = vec![];
        let current_parent = 0; // dunno
        let tree = Graph::new();
        let knn_index = RTree::new();

        let mut exp = MeshExplorer {
            d,
            boundary,
            margin,
            basis_vectors,
            path_queue,
            current_parent,
            tree,
            knn_index,
            adherer: None,
            adherer_f,
        };

        exp.add_child(root, None);

        exp
    }

    pub fn knn_index(&self) -> &RTree<GeomWithData<[f64; N], usize>> {
        &self.knn_index
    }

    fn select_parent(&mut self) -> Option<(Halfspace<N>, NodeID, SVector<f64, N>)> {
        while let Some((path_id, path_vector)) = self.path_queue.dequeue() {
            let hs = &self.boundary[path_id];
            let point = *hs.b + self.d * path_vector;

            if !self.check_overlap(point) {
                return Some((*hs, path_id, path_vector));
            }
        }

        None
    }

    fn add_child(&mut self, hs: Halfspace<N>, parent_id: Option<NodeIndex>) {
        let next_id = self.tree.add_node(hs);
        if let Some(pid) = parent_id {
            self.tree.add_edge(pid, next_id, ());
        }

        self.path_queue
            .extend(self.get_next_paths_from(next_id.index()));

        let b: [f64; N] = hs.b.into();

        self.knn_index.insert(KnnNode::new(b, next_id.index()));
    }

    fn get_next_paths_from(&self, id: NodeID) -> Vec<Path<N>> {
        let hs = &self.boundary[id];
        let next_paths = Self::create_cardinals(hs.n, self.basis_vectors)
            .iter()
            .map(|&v| (id, v))
            .collect();

        next_paths
    }

    pub fn create_cardinals(
        n: SVector<f64, N>,
        basis_vectors: OMatrix<f64, Const<N>, Const<N>>,
    ) -> Vec<SVector<f64, N>> {
        let align_vector: SVector<f64, N> = basis_vectors.column(0).into();
        let span = Span::new(n, align_vector);
        let angle = align_vector.angle(&n);

        let axes = if angle <= 1e-10 {
            basis_vectors
        } else {
            let rot = span.get_rotater()(angle);
            rot * basis_vectors
        };

        let mut cardinals = vec![];

        for i in 1..axes.ncols() {
            let column: SVector<f64, N> = axes.column(i).into();
            cardinals.push(column);
            cardinals.push(-column);
        }

        cardinals
    }

    fn check_overlap(&self, p: SVector<f64, N>) -> bool {
        let p = svector_to_array(p);

        if let Some(nearest) = self.knn_index.nearest_neighbor(&p) {
            array_distance(&p, nearest.geom()) < self.margin
        } else {
            false
        }
    }

    fn get_parent(&self, id: NodeIndex) -> Option<NodeIndex> {
        if let Some(edge) = self.tree.edges_directed(id, Incoming).next() {
            return Some(edge.source());
        }
        None
    }
}

impl<const N: usize, F: AdhererFactory<N>> Explorer<N, F> for MeshExplorer<N, F> {
    fn step<C: Classifier<N>>(&mut self, classifier: &mut C) -> Result<Option<Sample<N>>> {
        if self.adherer.is_none() {
            if let Some((hs, id, v)) = self.select_parent() {
                self.current_parent = id;
                self.adherer = Some(self.adherer_f.adhere_from(hs, v * self.d))
            }
        }

        let node = if let Some(ref mut adh) = self.adherer {
            match adh.sample_next(classifier) {
                Ok(result) => {
                    let sample = *result;

                    if let AdhererState::FoundBoundary(hs) = adh.get_state() {
                        let p = *hs.b + self.d * sample.into_inner();

                        if !self.check_overlap(p) {
                            self.boundary.push(hs);
                        
                            self.add_child(hs, Some(NodeIndex::new(self.current_parent)));
                            self.adherer = None
                        }
                    }

                    Ok(Some(sample))
                }
                Err(e) => Err(e),
            }
        } else {
            // Ends exploration
            Ok(None)
        };

        node.inspect_err(|_| self.adherer = None)
    }

    fn boundary(&self) -> &Vec<Halfspace<N>> {
        &self.boundary
    }

    fn boundary_count(&self) -> usize {
        self.boundary.len()
    }

    fn describe(&self) -> ExplorationStatus<N, F> {
        let mut expl_params = HashMap::new();
        expl_params.insert("d".to_string(), self.d);
        expl_params.insert("margin".to_string(), self.margin);

        ExplorationStatus::new(
            "Mesh Explorer",
            type_name::<F>(),
            expl_params,
            self.adherer_f,
            &self.boundary,
            None,
        )
    }
}

impl<const N: usize, F: AdhererFactory<N>> Backpropagation<N> for MeshExplorer<N, F> {
    fn backprop(&mut self, child_id: NodeIndex, margin: f64) {
        let parent_indx = if let Some(index) = self.get_parent(child_id) {
            index
        } else {
            return; // root, nothing to backprop
        };

        let parent = self.boundary[parent_indx.index()];

        let b: [f64; N] = parent.b.into();
        let rtree = &self.knn_index;
        let neighbors = rtree.nearest_neighbor_iter(&b);

        let mut n = SVector::zeros();
        let mut count = 0;

        for node in neighbors {
            if array_distance(&b, node.geom()) <= margin {
                n += self.boundary[node.data].n;
                count += 1;
            } else {
                break;
            }
        }

        n /= count as f64;
        self.boundary[parent_indx.index()] = Halfspace {
            b: parent.b,
            n: n.normalize(),
        }
    }
}
