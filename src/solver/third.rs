use super::{
    steiner_tree::{self},
    Policy, Strategy,
};
use crate::{
    common::grid::Coordinate, gaussean_process::GaussianPredictor, input::Input, map::MapState,
    solver::IncreasingPolicy, ChangeMinMax,
};
use itertools::{izip, Itertools};
use nalgebra::{DMatrix, DVector};
use std::collections::VecDeque;

pub struct ConnectionStrategy {
    paths: Vec<Vec<Coordinate>>,
    child_strategy: Box<dyn Strategy>,
    stage: usize,
}

impl ConnectionStrategy {
    pub fn new(input: &Input, map: &MapState) -> Self {
        map.dump_pred(input, 1000);

        let paths = steiner_tree::calc_steiner_tree_paths(input, map, 0.0);
        let child_strategy = Self::gen_child_strategy(0, map, &paths);
        Self {
            paths,
            child_strategy,
            stage: 0,
        }
    }

    fn gen_child_strategy(
        stage: usize,
        map: &MapState,
        paths: &Vec<Vec<Coordinate>>,
    ) -> Box<dyn Strategy> {
        let ret: Box<dyn Strategy> = if stage == 0 {
            Box::new(PreBoringChildStrategy::new(paths.clone()))
        } else if stage == 1 {
            Box::new(FullPathChildStrategy::new(paths.clone(), map))
        } else {
            unreachable!()
        };

        ret
    }
}

impl Strategy for ConnectionStrategy {
    fn get_next_policies(&mut self, input: &Input, map: &mut MapState) -> Vec<Box<dyn Policy>> {
        if self.child_strategy.is_completed() {
            self.stage += 1;
            self.child_strategy = Self::gen_child_strategy(self.stage, map, &self.paths);
        }

        self.child_strategy.get_next_policies(input, map)
    }

    fn is_completed(&self) -> bool {
        self.stage == 1 && self.child_strategy.is_completed()
    }
}

struct PreBoringChildStrategy {
    paths: Vec<Vec<Coordinate>>,
    is_completed: bool,
}

impl PreBoringChildStrategy {
    fn new(paths: Vec<Vec<Coordinate>>) -> Self {
        Self {
            paths,
            is_completed: false,
        }
    }

    fn gen_dp_policy(c: Coordinate, input: &Input, map: &MapState) -> DpPolicy {
        let mean = map.get_pred_sturdiness(c, 0.0) as f64;
        let stddev =
            (map.get_pred_sturdiness(c, 1.0) - map.get_pred_sturdiness(c, -1.0)) as f64 / 2.0;
        DpPolicy::new(input, c, mean, stddev * stddev)
    }
}

impl Strategy for PreBoringChildStrategy {
    fn get_next_policies(&mut self, input: &Input, map: &mut MapState) -> Vec<Box<dyn Policy>> {
        const STRIDE: usize = 10;
        let mut digged = map.digged.clone();
        let mut strategies: Vec<Box<dyn Policy>> = vec![];

        for path in self.paths.iter() {
            let mut last_index = 0;

            if !digged.is_digged(path[0]) {
                strategies.push(Box::new(Self::gen_dp_policy(path[0], input, map)));
                digged.dig(path[0]);
            }

            for i in 1..(path.len() - 1) {
                if !digged.is_digged(path[i]) {
                    if i - last_index >= STRIDE {
                        strategies.push(Box::new(Self::gen_dp_policy(path[i], input, map)));
                        digged.dig(path[i]);
                        last_index = i;
                    }
                } else {
                    last_index = i;
                }
            }

            if !digged.is_digged(path[path.len() - 1]) {
                strategies.push(Box::new(Self::gen_dp_policy(
                    path[path.len() - 1],
                    input,
                    map,
                )));
                digged.dig(path[path.len() - 1]);
            }
        }

        self.is_completed = true;

        strategies
    }

    fn is_completed(&self) -> bool {
        self.is_completed
    }
}

pub struct FullPathChildStrategy {
    is_completed: bool,
    paths: Vec<Vec<Coordinate>>,
    gausseans: Vec<GaussianPredictor>,
}

impl FullPathChildStrategy {
    pub fn new(paths: Vec<Vec<Coordinate>>, map: &MapState) -> Self {
        let gausseans = Self::get_gausseans(map, &paths);

        Self {
            is_completed: false,
            paths,
            gausseans,
        }
    }

    fn get_gausseans(map: &MapState, paths: &[Vec<Coordinate>]) -> Vec<GaussianPredictor> {
        let mut gausseans = vec![];

        for i in 0..paths.len() {
            let mut gaussean = GaussianPredictor::new();

            for (i, &c) in paths[i].iter().enumerate() {
                if map.digged.is_digged(c) {
                    // 事前にsqrtをかけておく
                    let x = DVector::from_vec(vec![i as f64]);
                    let y = (map.damages[c] as f64).sqrt();
                    gaussean.add_data(x, y);
                }
            }

            // パラメータをグリッドサーチ
            let t1_cands = (2..8).map(|v| 2.0f64.powi(v)).collect_vec();
            let t2_cands = (1..6)
                .map(|v| {
                    let v = 2.0f64.powi(v);
                    v * v
                })
                .collect_vec();
            let t3_cands = (0..5).map(|v| 2.0f64.powi(v)).collect_vec();
            gaussean.grid_search_theta(&t1_cands, &t2_cands, &t3_cands);

            gausseans.push(gaussean);
        }

        gausseans
    }

    fn predict(
        &mut self,
        path_id: usize,
        target_indices: &[usize],
        map: &MapState,
    ) -> Vec<(f64, f64, f64)> {
        if target_indices.len() == 0 {
            return vec![];
        }

        let gaussean = &mut self.gausseans[path_id];
        let path = &self.paths[path_id];
        gaussean.clear();

        for (i, &c) in path.iter().enumerate() {
            if map.digged.is_digged(c) {
                // 事前にsqrtをかけておく
                let x = DVector::from_vec(vec![i as f64]);
                let y = (map.damages[c] as f64).sqrt();
                gaussean.add_data(x, y);
            }
        }

        let mut x = vec![];

        for &i in target_indices.iter() {
            x.push(i as f64);
        }

        let x = DMatrix::from_row_slice(target_indices.len(), 1, &x);

        let (mut y_mean, mut y_var) = gaussean.gaussian_process_regression(&x);
        let mut y_lower = vec![];
        let mut y_upper = vec![];

        // sqrt(10), sqrt(5000)
        const LOWER: f64 = 3.16227766;
        const UPPER: f64 = 70.71067811;

        for (mean, var) in y_mean.iter_mut().zip(y_var.iter_mut()) {
            // TODO: たまに変な値になるので要調査
            var.change_max(1.0);
            mean.change_max(LOWER);
            mean.change_min(UPPER);
            y_lower.push((*mean - var.sqrt()).max(LOWER));
            y_upper.push((*mean + var.sqrt()).min(UPPER));
        }

        let mut result = vec![];

        for (l, m, h) in izip!(&y_lower, &y_mean, &y_upper) {
            result.push((l * l, m * m, h * h));
        }

        result
    }

    fn get_half_div_policies(
        &mut self,
        input: &Input,
        path_id: usize,
        map: &MapState,
    ) -> Vec<DpPolicy> {
        let digged_indices = Self::get_digged_indices(&self.paths[path_id], map);
        let target_points = Self::get_target_points(&self.paths[path_id], &digged_indices);
        let predictions = self.predict(path_id, &target_points, map);
        let path = &self.paths[path_id];

        let mut policies = vec![];

        for (&i, (lower, mean, upper)) in target_points.iter().zip(predictions.iter()) {
            // 本来は2乗した空間での正規分布を考える必要があるが、
            // めんどくさいので上側と下側（±σ）を2で割ったものを標準偏差としている
            let std_dev = (upper - lower) / 2.0;
            let policy = DpPolicy::new(input, path[i], *mean, std_dev * std_dev);
            policies.push(policy);
        }

        policies
    }

    fn get_digged_indices(path: &[Coordinate], map: &MapState) -> Vec<usize> {
        let mut digged_indices = vec![];

        for i in 0..path.len() {
            let c = path[i];
            if map.digged.is_digged(c) {
                digged_indices.push(i);
            }
        }

        digged_indices
    }

    fn get_target_points(path: &[Coordinate], digged_indices: &[usize]) -> Vec<usize> {
        // 必ず両端が掘られているので例外処理は不要
        assert!(digged_indices.len() >= 2);
        assert!(digged_indices[0] == 0);
        assert!(digged_indices[digged_indices.len() - 1] == path.len() - 1);
        let mut target_points = vec![];

        for i in 0..(digged_indices.len() - 1) {
            let target = (digged_indices[i] + digged_indices[i + 1]) / 2;

            if target != digged_indices[i] {
                target_points.push(target);
            }
        }

        target_points
    }
}

impl Strategy for FullPathChildStrategy {
    fn get_next_policies(
        &mut self,
        input: &crate::input::Input,
        map: &mut crate::map::MapState,
    ) -> Vec<Box<dyn super::Policy>> {
        let mut digged = map.digged.clone();
        let mut policies: Vec<Box<dyn super::Policy>> = vec![];

        for path_id in 0..self.paths.len() {
            for policy in self.get_half_div_policies(input, path_id, map) {
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
    pred_expected: f64,
    pred_std_dev: f64,
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
            pred_expected: expected,
            pred_std_dev: variance.sqrt(),
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

    fn comment(&self) -> Vec<String> {
        vec![
            format!("expected: {}", self.pred_expected),
            format!("stddev  : {}", self.pred_std_dev),
        ]
    }
}
