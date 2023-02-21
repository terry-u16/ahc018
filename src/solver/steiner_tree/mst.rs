use crate::{acl::dsu::Dsu, input::Input};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct Edge {
    packed: i64,
}

impl Edge {
    pub(super) fn new(cost: i32, i: usize, j: usize) -> Self {
        let packed = ((cost as i64) << 32) | ((i as i64) << 16) | (j as i64);
        Self { packed }
    }

    fn cost(&self) -> i32 {
        (self.packed >> 32) as i32
    }

    fn i(&self) -> usize {
        const MASK: usize = (1 << 16) - 1;
        ((self.packed >> 16) as usize) & MASK
    }

    fn j(&self) -> usize {
        const MASK: usize = (1 << 16) - 1;
        (self.packed as usize) & MASK
    }
}

pub(super) trait MstAggregator<T> {
    fn add(&mut self, i: usize, j: usize, cost: i32);
    fn get_result(self) -> T;
}

pub(super) struct MstEdgeAggregator {
    edges: Vec<Edge>,
}

impl MstEdgeAggregator {
    pub(super) fn new() -> Self {
        Self { edges: vec![] }
    }
}

impl MstAggregator<Vec<Edge>> for MstEdgeAggregator {
    fn add(&mut self, i: usize, j: usize, cost: i32) {
        self.edges.push(Edge::new(cost, i, j));
    }

    fn get_result(self) -> Vec<Edge> {
        self.edges
    }
}

pub(super) struct MstCostAggregator {
    cost: i32,
}

impl MstCostAggregator {
    pub(super) fn new() -> Self {
        Self { cost: 0 }
    }
}

impl MstAggregator<i32> for MstCostAggregator {
    fn add(&mut self, _i: usize, _j: usize, cost: i32) {
        self.cost += cost;
    }

    fn get_result(self) -> i32 {
        self.cost
    }
}

pub(super) fn calculate_mst<T>(
    input: &Input,
    n: usize,
    mut edges: Vec<Edge>,
    mut aggregator: impl MstAggregator<T>,
) -> T {
    edges.sort_unstable();

    let mut dsu = Dsu::new(n);

    // 水源は繋いでおく
    for i in 1..input.water_count {
        dsu.merge(0, i);
    }

    for edge in edges {
        let i = edge.i();
        let j = edge.j();
        let c = edge.cost();

        if !dsu.same(i, j) {
            dsu.merge(i, j);
            aggregator.add(i, j, c);
        }
    }

    aggregator.get_result()
}
