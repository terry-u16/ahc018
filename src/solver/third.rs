use super::{steiner_tree::calc_steiner_tree_paths, PredictedPolicy, Strategy};

pub struct FullPathStrategy {
    is_completed: bool,
}

impl FullPathStrategy {
    pub fn new() -> Self {
        Self {
            is_completed: false,
        }
    }
}

impl Strategy for FullPathStrategy {
    fn get_next_policies(
        &mut self,
        input: &crate::input::Input,
        map: &mut crate::map::MapState,
    ) -> Vec<Box<dyn super::Policy>> {
        map.update_prediction();
        let paths = calc_steiner_tree_paths(input, map, 1.2);
        let mut digged = map.digged.clone();
        let mut policies: Vec<Box<dyn super::Policy>> = vec![];
        map.dump_pred(input, 1000);

        for &c in paths.iter().flatten() {
            if !digged.is_digged(c) {
                digged.dig(c);
                policies.push(Box::new(PredictedPolicy::new(c, 1.2, input)));
            }
        }

        self.is_completed = true;

        policies
    }

    fn is_completed(&self) -> bool {
        self.is_completed
    }
}
