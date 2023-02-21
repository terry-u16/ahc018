use rand::Rng;
use rand_pcg::Pcg64Mcg;

use crate::{common::grid::Coordinate, input::Input, map::MapState};

use super::{Policy, Strategy};

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

    fn get_random_poicies(&self, input: &Input, map: &MapState) -> Vec<Box<dyn Policy>> {
        const KEEP_OUT_DIST: usize = 25;
        const TARGET_COUNT: usize = 50;
        let remain = TARGET_COUNT - (input.water_count + input.house_count);
        let mut rng = Pcg64Mcg::new(42);
        let mut policies: Vec<Box<dyn Policy>> = vec![];
        let mut digged = map.digged.clone();

        while policies.len() < remain {
            let row = rng.gen_range(0, input.map_size);
            let col = rng.gen_range(0, input.map_size);
            let c = Coordinate::new(row, col);
            if !digged.has_digged_nearby(c, KEEP_OUT_DIST) {
                policies.push(Box::new(IncreasingPolicy::new(c)));
                digged.dig(c);
            }
        }

        policies
    }
}

impl Strategy for RandomBoringStrategy {
    fn get_next_policies(
        &mut self,
        input: &crate::input::Input,
        map: &crate::map::MapState,
    ) -> Vec<Box<dyn super::Policy>> {
        let result = match self.stage {
            0 => self.get_water_poicies(input),
            1 => self.get_house_poicies(input),
            2 => self.get_random_poicies(input, map),
            _ => vec![],
        };

        self.stage += 1;
        result
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

    fn next_power(&mut self) -> i32 {
        const POWER_SERIES: [i32; 5] = [20, 30, 50, 100, 200];
        let result = POWER_SERIES[self.count.min(POWER_SERIES.len() - 1)];
        self.count += 1;
        result
    }
}
