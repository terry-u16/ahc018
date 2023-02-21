use crate::solver::IncreasingPolicy;

use super::{steiner_tree::calc_steiner_tree_paths, Strategy};

pub struct SkippingPathStrategy {
    iter: usize,
}

impl SkippingPathStrategy {
    pub fn new() -> Self {
        Self { iter: 0 }
    }
}

impl Strategy for SkippingPathStrategy {
    fn get_next_policies(
        &mut self,
        input: &crate::input::Input,
        map: &mut crate::map::MapState,
    ) -> Vec<Box<dyn super::Policy>> {
        const MAX_ITER: usize = 5;
        if self.iter >= MAX_ITER {
            return vec![];
        }

        self.iter += 1;
        let near_threshold = if self.iter <= 3 { 20 } else { 10 };

        map.update_prediction();
        let paths = calc_steiner_tree_paths(input, map, 1.5);
        let mut digged = map.digged.clone();
        let mut policies: Vec<Box<dyn super::Policy>> = vec![];

        for &c in paths.iter().flatten() {
            if !digged.has_digged_nearby(c, near_threshold) {
                digged.dig(c);
                policies.push(Box::new(IncreasingPolicy::new(c)));
            }
        }

        policies
    }
}
