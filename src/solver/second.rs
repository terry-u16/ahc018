use std::cmp::Reverse;

use itertools::Itertools;

use crate::solver::IncreasingPolicy;

use super::{steiner_tree::calc_steiner_tree_paths, Strategy};

pub struct SkippingPathStrategy {
    iter: usize,
    is_completed: bool,
}

impl SkippingPathStrategy {
    pub fn new() -> Self {
        Self {
            iter: 0,
            is_completed: false,
        }
    }
}

impl Strategy for SkippingPathStrategy {
    fn get_next_policies(
        &mut self,
        input: &crate::input::Input,
        map: &mut crate::map::MapState,
    ) -> Vec<Box<dyn super::Policy>> {
        const DIST_SERIES: [usize; 10] = [30, 27, 24, 21, 18, 15, 15, 15, 15, 15];
        if self.iter >= DIST_SERIES.len() {
            self.is_completed = true;
            return vec![];
        }

        let near_threshold = DIST_SERIES[self.iter];
        self.iter += 1;

        map.update_prediction(&input);

        let paths = calc_steiner_tree_paths(input, map, -0.3);
        let mut candidates = paths.iter().flatten().copied().collect_vec();
        candidates.sort_by_cached_key(|&c| {
            let stddev = map.get_pred_sturdiness(c, 0.0) - map.get_pred_sturdiness(c, -1.0);
            Reverse(stddev)
        });

        let mut digged = map.digged.clone();
        let mut policies: Vec<Box<dyn super::Policy>> = vec![];

        for &c in candidates.iter() {
            if !digged.has_digged_nearby(c, near_threshold) {
                digged.dig(c);
                policies.push(Box::new(IncreasingPolicy::new(c)));
            }
        }

        policies
    }

    fn is_completed(&self) -> bool {
        self.is_completed
    }
}
