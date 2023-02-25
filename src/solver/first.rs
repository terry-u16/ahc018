use super::{IncreasingPolicy, Policy, Strategy};
use crate::input::Input;

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
