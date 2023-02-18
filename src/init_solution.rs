use std::collections::VecDeque;

use crate::{
    common::grid::{Coordinate, Map2d, ADJACENTS},
    input::Input,
    output::{Action, DiggingResult},
    ChangeMinMax,
};

#[derive(Debug, Clone)]
pub struct InitSolver {
    digged: Map2d<bool>,
}

impl InitSolver {
    pub fn new(input: &Input) -> Self {
        let digged = Map2d::new(vec![false; input.map_size * input.map_size], input.map_size);
        Self { digged }
    }

    pub fn get_next_action(&mut self, input: &Input) -> Action {
        let house_index = self.get_not_connected_houses(input)[0];
        let next_pos = self.bfs01(input.houses[house_index], input);
        Action::new(next_pos, 100)
    }

    pub fn update(&mut self, result: DiggingResult) {
        if let DiggingResult::Broken(action) = result {
            self.digged[action.coordinate] = true;
        }
    }

    fn get_not_connected_houses(&self, input: &Input) -> Vec<usize> {
        let mut results = vec![];

        for i in 0..input.house_count {
            let mut seen = Map2d::new(vec![false; input.map_size * input.map_size], input.map_size);

            let depth = 0;

            if !self.dfs(input.houses[i], &mut seen, input, depth) {
                results.push(i);
            }
        }

        results
    }

    fn dfs(&self, c: Coordinate, seen: &mut Map2d<bool>, input: &Input, depth: usize) -> bool {
        seen[c] = true;

        if input.waters.iter().any(|w| *w == c) {
            return true;
        }

        if !self.digged[c] {
            return false;
        }

        for adj in ADJACENTS.iter() {
            let next = c + *adj;

            if !next.in_map(input.map_size) || seen[next] {
                continue;
            }

            if self.dfs(next, seen, input, depth + 1) {
                return true;
            }
        }

        false
    }

    fn bfs01(&self, house: Coordinate, input: &Input) -> Coordinate {
        const INF: i32 = std::i32::MAX / 2;
        let mut dists = Map2d::new(vec![INF; input.map_size * input.map_size], input.map_size);
        let mut from = Map2d::new(
            vec![Coordinate::new(!0, !0); input.map_size * input.map_size],
            input.map_size,
        );
        let mut queue = VecDeque::new();
        dists[house] = 0;
        queue.push_back(house);

        while let Some(c) = queue.pop_front() {
            for adj in ADJACENTS.iter() {
                let next = c + *adj;

                if !next.in_map(input.map_size) || dists[next] < INF {
                    continue;
                }

                from[next] = c;

                if self.digged[next] {
                    dists[next] = dists[c];
                    queue.push_front(next);
                } else {
                    dists[next] = dists[c] + 1;
                    queue.push_back(next);
                }
            }
        }

        let mut current = Coordinate::new(!0, !0);
        let mut nearest_dist = std::i32::MAX;

        for water in input.waters.iter() {
            if nearest_dist.change_min(dists[*water]) {
                current = *water;
            }
        }

        loop {
            if !self.digged[current] {
                return current;
            }

            current = from[current];
        }
    }
}
