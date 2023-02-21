mod first;
mod second;
mod steiner_tree;
mod third;

use std::collections::VecDeque;

use crate::{
    common::grid::Coordinate,
    input::Input,
    map::MapState,
    output::{Action, DiggingResult},
    solver::steiner_tree::calc_steiner_tree_paths,
};

use self::{first::RandomBoringStrategy, second::SkippingPathStrategy, third::FullPathStrategy};

pub struct Solver<'a> {
    input: &'a Input,
    map: MapState,
    strategies: Vec<Box<dyn Strategy>>,
    policies: VecDeque<Box<dyn Policy>>,
    stage: usize,
}

impl<'a> Solver<'a> {
    pub fn new(input: &'a Input) -> Self {
        let map = MapState::new(input);
        let strategies: Vec<Box<dyn Strategy>> = vec![
            Box::new(RandomBoringStrategy::new()),
            Box::new(SkippingPathStrategy::new()),
            Box::new(FullPathStrategy::new()),
        ];

        Self {
            input,
            map,
            strategies,
            policies: VecDeque::new(),
            stage: 0,
        }
    }

    pub fn get_next_action(&mut self) -> Action {
        while self.policies.len() == 0 {
            let policies = self.strategies[self.stage].get_next_policies(self.input, &mut self.map);

            if policies.len() == 0 {
                // 途中経過を出力
                self.map.update_prediction();
                self.map.dump_pred(self.input, 1000);
                eprintln!();
                /*
                    let mut digged = self.map.digged.clone();
                    const SAFETY_FACTOR: f64 = 1.2;

                    for path in calc_steiner_tree_paths(self.input, &self.map, SAFETY_FACTOR) {
                        for &c in path.iter() {
                            if !digged.is_digged(c) {
                                digged.dig(c);
                                self.policies.push_back(Box::new(IncreasingPolicy::new(c)));
                            }
                        }
                    }
                */
                self.stage += 1;
                continue;
            }

            for policy in policies {
                self.policies.push_back(policy);
            }
        }

        let policy = self.policies.front_mut().unwrap();
        Action::new(policy.target(), policy.next_power())
    }

    pub fn update(&mut self, result: DiggingResult) {
        let (action, broken) = match result {
            DiggingResult::NotBroken(action) => (action, false),
            DiggingResult::Broken(action) => (action, true),
            DiggingResult::Completed => unreachable!(),
        };

        self.map.dig(action.coordinate, action.power, broken);

        if broken {
            self.policies.pop_front();
        }
    }
}

trait Strategy {
    fn get_next_policies(&mut self, input: &Input, map: &mut MapState) -> Vec<Box<dyn Policy>>;
}

trait Policy {
    fn target(&self) -> Coordinate;
    fn next_power(&mut self) -> i32;
}

struct IncreasingPolicy {
    count: usize,
    target: Coordinate,
}

impl IncreasingPolicy {
    fn new(target: Coordinate) -> Self {
        Self { count: 0, target }
    }
}

impl Policy for IncreasingPolicy {
    fn target(&self) -> Coordinate {
        self.target
    }

    fn next_power(&mut self) -> i32 {
        const POWER_SERIES: [i32; 5] = [20, 30, 50, 100, 200];
        let result = POWER_SERIES[self.count.min(POWER_SERIES.len() - 1)];
        self.count += 1;
        result
    }
}
