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
        while let Some(p) = self.policies.front() {
            if p.give_up() {
                self.map.digged.mark_revealed(p.target());
                self.policies.pop_front();
            } else {
                break;
            }
        }

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
        Action::new(
            policy.target(),
            policy.next_power(&self.map),
            policy.comment(),
        )
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
    fn give_up(&self) -> bool;
    fn comment(&self) -> Vec<String>;
}
