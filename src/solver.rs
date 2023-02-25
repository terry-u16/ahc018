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
};

use self::{first::RandomBoringStrategy, second::SkippingPathStrategy, third::ConnectionStrategy};

pub struct Solver<'a> {
    input: &'a Input,
    map: MapState,
    strategy: Box<dyn Strategy>,
    policies: VecDeque<Box<dyn Policy>>,
    stage: usize,
}

impl<'a> Solver<'a> {
    pub fn new(input: &'a Input) -> Self {
        let map = MapState::new(input);
        let strategy = Self::gen_strategy(0, input, &map);

        Self {
            input,
            map,
            strategy,
            policies: VecDeque::new(),
            stage: 0,
        }
    }

    pub fn get_next_action(&mut self) -> Action {
        while self.policies.len() == 0 {
            while self.strategy.is_completed() {
                self.stage += 1;
                self.strategy = Self::gen_strategy(self.stage, &self.input, &self.map);
            }

            let policies = self.strategy.get_next_policies(self.input, &mut self.map);

            for policy in policies {
                self.policies.push_back(policy);
            }
        }

        let policy = self.policies.front_mut().unwrap();
        Action::new(policy.target(), policy.next_power(&self.map))
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

    fn gen_strategy(stage: usize, input: &Input, map: &MapState) -> Box<dyn Strategy> {
        let ret: Box<dyn Strategy> = if stage == 0 {
            Box::new(RandomBoringStrategy::new())
        } else if stage == 1 {
            Box::new(SkippingPathStrategy::new())
        } else if stage == 2 {
            Box::new(ConnectionStrategy::new(input, map))
        } else {
            unreachable!()
        };

        ret
    }
}

trait Strategy {
    fn get_next_policies(&mut self, input: &Input, map: &mut MapState) -> Vec<Box<dyn Policy>>;
    fn is_completed(&self) -> bool;
}

trait Policy {
    fn target(&self) -> Coordinate;
    fn next_power(&mut self, map: &MapState) -> i32;
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

    fn next_power(&mut self, _map: &MapState) -> i32 {
        const POWER_SERIES: [i32; 5] = [20, 30, 50, 100, 200];
        let result = POWER_SERIES[self.count.min(POWER_SERIES.len() - 1)];
        self.count += 1;
        result
    }
}

struct PredictedPolicy {
    target: Coordinate,
    safety_factor: f64,
    max: i32,
}

impl PredictedPolicy {
    fn new(target: Coordinate, safety_factor: f64, input: &Input) -> Self {
        Self {
            target,
            safety_factor,
            max: input.exhausting_energy * 10,
        }
    }
}

impl Policy for PredictedPolicy {
    fn target(&self) -> Coordinate {
        self.target
    }

    fn next_power(&mut self, map: &MapState) -> i32 {
        map.get_pred_sturdiness(self.target, self.safety_factor)
            .min(self.max)
    }
}
