use std::collections::VecDeque;

use crate::{
    common::grid::Coordinate,
    input::Input,
    map::MapState,
    output::{Action, DiggingResult},
};

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
        let strategies = vec![];

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
            let policies = self.strategies[self.stage].get_next_policies(self.input, &self.map);

            if policies.len() == 0 {
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
    fn get_next_policies(&mut self, input: &Input, map: &MapState) -> Vec<Box<dyn Policy>>;
}

trait Policy {
    fn target(&self) -> Coordinate;
    fn next_power(&mut self) -> i32;
}
