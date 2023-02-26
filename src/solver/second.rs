use std::cmp::Reverse;

use itertools::Itertools;

use crate::{common::grid::Coordinate, map::MapState};

use super::{steiner_tree::calc_steiner_tree_paths, Policy, Strategy};

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
        const DIST_SERIES: [usize; 10] = [30, 27, 24, 21, 18, 15, 12, 10, 10, 10];
        const SIGMA_SERIES: [f64; 10] =
            [-0.3, -0.3, -0.25, -0.25, -0.2, -0.2, -0.15, -0.15, -0.1, 0.0];
        if self.iter >= DIST_SERIES.len() {
            self.is_completed = true;
            return vec![];
        }

        let near_threshold = DIST_SERIES[self.iter];

        map.update_prediction(&input);

        let paths = calc_steiner_tree_paths(input, map, SIGMA_SERIES[self.iter]);
        let mut candidates = paths.iter().flatten().copied().collect_vec();
        candidates.sort_by_cached_key(|&c| {
            let stddev = map.get_pred_sturdiness(c, 0.0) - map.get_pred_sturdiness(c, -1.0);
            Reverse(stddev)
        });

        let mut digged = map.digged.clone();
        let mut policies: Vec<Box<dyn super::Policy>> = vec![];

        for &c in candidates.iter() {
            if !digged.has_revealed_nearby(c, near_threshold) {
                digged.dig(c);
                policies.push(Box::new(IncreasingPolicy::new(c)));
            }
        }

        self.iter += 1;

        policies
    }

    fn is_completed(&self) -> bool {
        self.is_completed
    }
}

struct IncreasingPolicy {
    count: usize,
    target: Coordinate,
    total_damage: i32,
}

impl IncreasingPolicy {
    fn new(target: Coordinate) -> Self {
        Self {
            count: 0,
            target,
            total_damage: 0,
        }
    }
}

impl Policy for IncreasingPolicy {
    fn target(&self) -> Coordinate {
        self.target
    }

    fn next_power(&mut self, _map: &MapState) -> i32 {
        const POWER_SERIES: [i32; 3] = [50, 150, 200];
        let result = POWER_SERIES[self.count.min(POWER_SERIES.len() - 1)];
        self.count += 1;
        self.total_damage += result;
        result
    }

    fn give_up(&self) -> bool {
        self.total_damage >= 1000
    }

    fn comment(&self) -> Vec<String> {
        vec![]
    }
}
