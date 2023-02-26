use super::{Policy, Strategy};
use crate::{common::grid::Coordinate, input::Input, map::MapState};

pub struct RandomBoringStrategy {
    stage: usize,
}

impl RandomBoringStrategy {
    pub fn new() -> Self {
        Self { stage: 0 }
    }

    fn get_water_poicies(&self, input: &Input) -> Vec<Box<dyn Policy>> {
        let mut policies = vec![];

        for w in input.waters.iter() {
            let p: Box<dyn Policy> = Box::new(IncreasingPolicy::new(*w));
            policies.push(p);
        }

        policies
    }

    fn get_house_poicies(&self, input: &Input) -> Vec<Box<dyn Policy>> {
        let mut policies = vec![];

        for h in input.houses.iter() {
            let p: Box<dyn Policy> = Box::new(IncreasingPolicy::new(*h));
            policies.push(p);
        }

        policies
    }
}

impl Strategy for RandomBoringStrategy {
    fn get_next_policies(
        &mut self,
        input: &crate::input::Input,
        _map: &mut crate::map::MapState,
    ) -> Vec<Box<dyn super::Policy>> {
        let result = match self.stage {
            0 => self.get_water_poicies(input),
            1 => self.get_house_poicies(input),
            _ => vec![],
        };

        self.stage += 1;
        result
    }

    fn is_completed(&self) -> bool {
        self.stage == 3
    }
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

    fn give_up(&mut self) -> bool {
        false
    }

    fn comment(&self) -> Vec<String> {
        vec![]
    }

    fn cancelled(&self) -> bool {
        false
    }
}
