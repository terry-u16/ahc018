use super::{steiner_tree::calc_steiner_tree_paths, Policy, PredictedPolicy, Strategy};
use crate::{common::grid::Coordinate, input::Input, map::MapState, ChangeMinMax};
use itertools::Itertools;
use std::collections::VecDeque;

pub struct FullPathStrategy {
    is_initialized: bool,
    is_completed: bool,
    paths: Vec<Vec<Coordinate>>,
}

impl FullPathStrategy {
    pub fn new() -> Self {
        Self {
            is_completed: false,
            is_initialized: false,
            paths: vec![],
        }
    }

    fn initialize(&mut self, input: &Input, map: &mut MapState) {
        map.update_prediction();
        let paths = calc_steiner_tree_paths(input, map, 1.2);
        map.dump_pred(input, 1000);

        self.is_initialized = true;
        self.paths = paths;
    }

    fn get_half_div_policies(
        &self,
        input: &Input,
        path: &[Coordinate],
        map: &MapState,
    ) -> Vec<DpPolicy> {
        let digged_indices = Self::get_digged_indices(path, map);
        let variance = Self::get_variance(path, &digged_indices, map);
        let target_points = Self::get_target_points(path, &digged_indices);

        let mut policies = vec![];

        for c in target_points.iter() {
            let expected = map.get_pred_sturdiness(*c, 1.0) as f64;
            let policy = DpPolicy::new(input, *c, expected, variance);
            policies.push(policy);
        }

        policies
    }

    fn get_digged_indices(path: &[Coordinate], map: &MapState) -> Vec<usize> {
        let mut digged_indices = vec![];

        for i in 1..(path.len() - 1) {
            let c = path[i];
            if map.digged.is_digged(c) {
                digged_indices.push(i);
            }
        }

        digged_indices
    }

    fn get_variance(path: &[Coordinate], digged_indices: &[usize], map: &MapState) -> f64 {
        // 掘られてなかったら適当に0を返す
        if digged_indices.len() == 0 {
            return 100.0;
        }

        let mut variance = 0;

        for &i in digged_indices.iter() {
            // 予測との2乗誤差を計算する
            let c = path[i];
            let pred_diff = map.get_pred_sturdiness(c, 1.0) - map.damages[c];
            variance += pred_diff * pred_diff;
        }

        // 0になると面倒なので適当に1を足す
        (variance + 1) as f64 / digged_indices.len() as f64
    }

    fn get_target_points(path: &[Coordinate], digged_indices: &[usize]) -> Vec<Coordinate> {
        let mut target_points = vec![];

        // 端は掘られているということにしてしまう
        let mut digged = vec![];
        digged.push(0);
        for &i in digged_indices.iter() {
            digged.push(i);
        }
        digged.push(path.len() - 1);

        for i in 0..(digged.len() - 1) {
            let target = (digged[i] + digged[i + 1]) / 2;

            if target != digged[i] {
                target_points.push(path[target]);
            }
        }

        // 端が掘られていないケースがあるので追加する
        if digged_indices.len() == 0 || digged_indices[0] != 0 {
            target_points.push(path[0]);
        }

        if digged_indices.len() == 0 || digged_indices[digged_indices.len() - 1] != path.len() - 1 {
            target_points.push(path[path.len() - 1]);
        }

        target_points
    }
}

impl Strategy for FullPathStrategy {
    fn get_next_policies(
        &mut self,
        input: &crate::input::Input,
        map: &mut crate::map::MapState,
    ) -> Vec<Box<dyn super::Policy>> {
        if !self.is_initialized {
            self.initialize(input, map);
        }

        map.update_prediction();

        let mut digged = map.digged.clone();
        let mut policies: Vec<Box<dyn super::Policy>> = vec![];

        for path in self.paths.iter() {
            for policy in self.get_half_div_policies(input, path, map) {
                if !digged.is_digged(policy.coordinate) {
                    digged.dig(policy.coordinate);
                    policies.push(Box::new(policy));
                }
            }
        }

        if policies.len() == 0 {
            self.is_completed = true;
        }

        policies
    }

    fn is_completed(&self) -> bool {
        self.is_completed
    }
}

struct DpPolicy {
    coordinate: Coordinate,
    tasks: VecDeque<i32>,
    emergency_power: i32,
}

impl DpPolicy {
    fn new(input: &Input, c: Coordinate, expected: f64, variance: f64) -> Self {
        let sturdiness = Self::get_sturdiness_points(expected, variance);
        let cumulative_dist = Self::get_cumulative_dist(expected, variance, &sturdiness);
        let task_queue = Self::calc_power_dp(input, &sturdiness, &cumulative_dist);

        // 3σやってもダメだったら適当にやる
        let emergency_power = input.exhausting_energy * 5;

        Self {
            coordinate: c,
            tasks: task_queue,
            emergency_power,
        }
    }

    fn get_sturdiness_points(expected: f64, variance: f64) -> Vec<i32> {
        let std_dev = variance.sqrt();

        // 3σくらいまで見る
        let upper_bound = ((expected + 3.0 * std_dev).round() as usize).min(5000);

        const MAX_SIZE: usize = 500;

        if upper_bound <= MAX_SIZE {
            return (0..=(upper_bound as i32)).collect_vec();
        }

        let mut points = Vec::with_capacity(MAX_SIZE + 1);
        let stride = upper_bound as f64 / MAX_SIZE as f64;

        for i in 0..=MAX_SIZE {
            points.push((stride * i as f64).round() as i32);
        }

        points
    }

    /// 正規分布の累積分布を計算する
    fn get_cumulative_dist(expected: f64, variance: f64, sturdinesses: &[i32]) -> Vec<f64> {
        let mut center = 0;

        for i in 0..=sturdinesses.len() {
            if (sturdinesses[i] as f64) < expected {
                center = i;
            } else {
                break;
            }
        }

        // 真ん中から左方向に数値積分
        // 台形公式で十分やろ
        let mut cumulative_dists = vec![0.0; sturdinesses.len()];
        let mut prev_point = expected;
        let mut cumulative_dist = 0.5;
        let mut right = Self::normal_prob_dist(expected, variance, prev_point);
        for i in (0..=center).rev() {
            let x = sturdinesses[i] as f64;
            let d = prev_point - x;
            let left = Self::normal_prob_dist(expected, variance, x);
            cumulative_dist -= (left + right) * d / 2.0;
            cumulative_dists[i] = cumulative_dist;
            right = left;
            prev_point = x;
        }

        // 右方向
        let mut prev_point = expected;
        let mut cumulative_dist = 0.5;
        let mut left = Self::normal_prob_dist(expected, variance, prev_point);
        for i in (center + 1)..sturdinesses.len() {
            let x = sturdinesses[i] as f64;
            let d = x - prev_point;
            let right = Self::normal_prob_dist(expected, variance, x);
            cumulative_dist += (left + right) * d / 2.0;
            cumulative_dists[i] = cumulative_dist;
            left = right;
            prev_point = x;
        }

        // 分布は0未満の領域にも伸びるが、無視
        cumulative_dists
    }

    /// 正規分布の確率密度関数
    fn normal_prob_dist(expected: f64, variance: f64, x: f64) -> f64 {
        let diff = x - expected;
        1.0 / (2.0 * std::f64::consts::PI * variance).sqrt()
            * (-diff * diff / (2.0 * variance)).exp()
    }

    fn calc_power_dp(input: &Input, sturdiness: &[i32], cumulative_dist: &[f64]) -> VecDeque<i32> {
        let mut dp = vec![std::f64::MAX / 2.0; sturdiness.len()];
        let mut from = vec![0; sturdiness.len()];
        dp[0] = 0.0;

        for i in 0..dp.len() {
            for j in (i + 1)..dp.len() {
                let power = sturdiness[j] - sturdiness[i] + input.exhausting_energy;
                let next = dp[i] + (1.0 - cumulative_dist[i]) * power as f64;

                if dp[j].change_min(next) {
                    from[j] = i;
                }
            }
        }

        let mut queue = VecDeque::new();
        let mut current = sturdiness.len() - 1;

        while current != 0 {
            let next = from[current];
            let consumption = sturdiness[current] - sturdiness[next];
            queue.push_front(consumption);
            current = next;
        }

        queue
    }
}

impl Policy for DpPolicy {
    fn target(&self) -> Coordinate {
        self.coordinate
    }

    fn next_power(&mut self, _map: &MapState) -> i32 {
        if let Some(power) = self.tasks.pop_front() {
            power
        } else {
            self.emergency_power
        }
    }
}
