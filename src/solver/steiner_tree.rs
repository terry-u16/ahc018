mod mst;

use crate::{
    common::grid::{Coordinate, Map2d, ADJACENTS},
    input::Input,
    map::MapState,
    ChangeMinMax,
};
use itertools::Itertools;
use rand::{seq::SliceRandom, Rng};
use rand_pcg::Pcg64Mcg;
use std::{cmp::Reverse, collections::BinaryHeap};

use self::mst::{Edge, MstEdgeAggregator};

#[derive(Debug, Clone)]
struct Environment<'a> {
    fixed_positions: Vec<Coordinate>,
    fixed_edges: Vec<Edge>,
    dists: Vec<Map2d<i32>>,
    input: &'a Input,
}

impl<'a> Environment<'a> {
    fn new(input: &'a Input, map: &MapState, safety_factor: f64) -> Self {
        let fixed_positions = Self::get_fixed_positions(input);
        let dists = Self::get_dists(&fixed_positions, input, map, safety_factor);

        let mut edges = vec![];

        for i in 0..fixed_positions.len() {
            for j in (i + 1)..fixed_positions.len() {
                edges.push(Edge::new(dists[i][fixed_positions[j]], i, j));
            }
        }

        let fixed_edges = mst::calculate_mst(
            input,
            fixed_positions.len(),
            edges,
            mst::MstEdgeAggregator::new(),
        );

        Self {
            fixed_positions,
            fixed_edges,
            dists,
            input,
        }
    }

    fn get_fixed_positions(input: &Input) -> Vec<Coordinate> {
        input
            .waters
            .iter()
            .copied()
            .chain(input.houses.iter().copied())
            .collect()
    }

    fn get_dists(
        positions: &[Coordinate],
        input: &Input,
        map: &MapState,
        safety_factor: f64,
    ) -> Vec<Map2d<i32>> {
        const INF: i32 = std::i32::MAX / 2;
        let n = input.map_size;
        let mut all_dists = vec![];

        for i in 0..positions.len() {
            let mut distances = Map2d::new(vec![INF; n * n], n);
            let mut queue = BinaryHeap::new();
            distances[positions[i]] = 0;
            queue.push(Reverse((0, positions[i])));

            while let Some(Reverse((dist, c))) = queue.pop() {
                if distances[c] < dist {
                    continue;
                }

                for adj in ADJACENTS.iter() {
                    let next = c + *adj;

                    if next.in_map(n) {
                        let added_cost = if map.digged.is_digged(next) {
                            0
                        } else {
                            map.get_pred_cost(next, safety_factor, input)
                        };
                        let next_dist = dist + added_cost;

                        if distances[next].change_min(next_dist) {
                            queue.push(Reverse((next_dist, next)));
                        }
                    }
                }
            }

            all_dists.push(distances);
        }

        all_dists
    }
}

#[derive(Debug, Clone)]
struct State {
    waypoints: Vec<Coordinate>,
}

impl State {
    fn new() -> Self {
        Self { waypoints: vec![] }
    }

    fn calc_score(&self, env: &Environment) -> i32 {
        let edges = Self::get_edge_candidates(&self, env);
        let n = env.fixed_positions.len() + self.waypoints.len();
        let cost = mst::calculate_mst(&env.input, n, edges, mst::MstCostAggregator::new());
        cost
    }

    fn get_edge_candidates(&self, env: &Environment) -> Vec<Edge> {
        let mut edges = env.fixed_edges.clone();

        // 追加頂点への辺を加える
        for i in 0..env.fixed_positions.len() {
            for (j, q) in self.waypoints.iter().enumerate() {
                let dist = env.dists[i][*q];
                edges.push(Edge::new(dist, i, env.fixed_positions.len() + j));
            }
        }

        edges
    }
}

trait Action {
    fn apply(&self, env: &Environment, state: &mut State);
    fn rollback(&self, env: &Environment, state: &mut State);
}

struct AddWaypoint {
    coordinate: Coordinate,
}

impl AddWaypoint {
    fn generate(env: &Environment, state: &State, rng: &mut Pcg64Mcg) -> Option<AddWaypoint> {
        let row = rng.gen_range(0, env.input.map_size);
        let col = rng.gen_range(0, env.input.map_size);
        let c = Coordinate::new(row, col);

        if !state.waypoints.contains(&c) {
            Some(AddWaypoint { coordinate: c })
        } else {
            None
        }
    }
}

impl Action for AddWaypoint {
    fn apply(&self, _env: &Environment, state: &mut State) {
        state.waypoints.push(self.coordinate);
    }

    fn rollback(&self, _env: &Environment, state: &mut State) {
        state.waypoints.pop();
    }
}

struct RemoveWaypoint {
    waypoint: Coordinate,
}

impl RemoveWaypoint {
    fn generate(_env: &Environment, state: &State, rng: &mut Pcg64Mcg) -> Option<RemoveWaypoint> {
        state
            .waypoints
            .choose(rng)
            .map(|c| RemoveWaypoint { waypoint: *c })
    }
}

impl Action for RemoveWaypoint {
    fn apply(&self, _env: &Environment, state: &mut State) {
        state.waypoints.retain(|c| *c != self.waypoint)
    }

    fn rollback(&self, _env: &Environment, state: &mut State) {
        state.waypoints.push(self.waypoint);
    }
}

struct MoveWaypoint {
    from: Coordinate,
    to: Coordinate,
}

impl MoveWaypoint {
    fn generate(env: &Environment, state: &State, rng: &mut Pcg64Mcg) -> Option<MoveWaypoint> {
        fn gen_diff(rng: &mut Pcg64Mcg) -> f64 {
            // ±[0.0, 16.0) からランダムに選択
            const MIN_POW2: f64 = 0.0;
            const MAX_POW2: f64 = 4.0;

            let sign = if rng.gen_bool(0.5) { 1.0 } else { -1.0 };
            let diff = 2.0f64.powf(rng.gen_range(MIN_POW2, MAX_POW2)) - 1.0;
            sign * diff
        }

        if let Some(c) = state.waypoints.choose(rng).copied() {
            let upper = (env.input.map_size - 1) as f64;
            let dr = gen_diff(rng);
            let dc = gen_diff(rng);
            let row = (c.row as f64 + dr).round().max(0.0).min(upper) as usize;
            let col = (c.col as f64 + dc).round().max(0.0).min(upper) as usize;
            let next_c = Coordinate::new(row, col);

            if state.waypoints.contains(&next_c) {
                None
            } else {
                Some(MoveWaypoint {
                    from: c,
                    to: next_c,
                })
            }
        } else {
            None
        }
    }
}

impl Action for MoveWaypoint {
    fn apply(&self, _env: &Environment, state: &mut State) {
        state.waypoints.retain(|c| *c != self.from);
        state.waypoints.push(self.to);
    }

    fn rollback(&self, _env: &Environment, state: &mut State) {
        state.waypoints.retain(|c| *c != self.to);
        state.waypoints.push(self.from);
    }
}

struct NoOp;

impl Action for NoOp {
    fn apply(&self, _env: &Environment, _state: &mut State) {
        // do nothing
    }

    fn rollback(&self, _env: &Environment, _state: &mut State) {
        // do nothing
    }
}

fn generate_action(env: &Environment, state: &State, rng: &mut Pcg64Mcg) -> Box<dyn Action> {
    for _ in 0..100 {
        let action_type = rng.gen_range(0, 4);

        if action_type == 0 {
            if let Some(act) = AddWaypoint::generate(env, state, rng) {
                return Box::new(act);
            }
        } else if action_type == 1 {
            if let Some(act) = RemoveWaypoint::generate(env, state, rng) {
                return Box::new(act);
            }
        } else {
            if let Some(act) = MoveWaypoint::generate(env, state, rng) {
                return Box::new(act);
            }
        }
    }

    Box::new(NoOp)
}

pub fn calc_steiner_tree_paths(
    input: &Input,
    map: &MapState,
    safety_factor: f64,
) -> Vec<Vec<Coordinate>> {
    let env = Environment::new(input, map, safety_factor);
    let state = State::new();
    let state = annealing(&env, state, 0.05);
    restore_steiner_paths(&env, &state, map, safety_factor)
}

fn annealing(env: &Environment, initial_state: State, duration: f64) -> State {
    let mut state = initial_state;
    let mut best_state = state.clone();
    let mut current_score = state.calc_score(env);
    let mut best_score = current_score;
    let init_score = current_score;

    let mut all_iter = 0;
    let mut valid_iter = 0;
    let mut accepted_count = 0;
    let mut update_count = 0;
    let mut rng = rand_pcg::Pcg64Mcg::new(42);

    let duration_inv = 1.0 / duration;
    let since = std::time::Instant::now();

    let temp0 = 5e1;
    let temp1 = 1e0;
    let mut inv_temp = 1.0 / temp0;

    loop {
        all_iter += 1;
        if (all_iter & ((1 << 4) - 1)) == 0 {
            let time = (std::time::Instant::now() - since).as_secs_f64() * duration_inv;
            let temp = f64::powf(temp0, 1.0 - time) * f64::powf(temp1, time);
            inv_temp = 1.0 / temp;

            if time >= 1.0 {
                break;
            }
        }

        // 変形
        let action = generate_action(env, &state, &mut rng);
        action.apply(env, &mut state);

        // スコア計算
        let new_score = state.calc_score(env);
        let score_diff = new_score - current_score;

        if score_diff <= 0 || rng.gen_bool(f64::exp(-score_diff as f64 * inv_temp)) {
            // 解の更新
            current_score = new_score;
            accepted_count += 1;

            if best_score.change_min(current_score) {
                best_state = state.clone();
                update_count += 1;
            }
        } else {
            action.rollback(env, &mut state);
        }

        valid_iter += 1;
    }

    eprintln!("===== annealing =====");
    eprintln!("init score : {}", init_score);
    eprintln!("score      : {}", best_score);
    eprintln!("all iter   : {}", all_iter);
    eprintln!("valid iter : {}", valid_iter);
    eprintln!("accepted   : {}", accepted_count);
    eprintln!("updated    : {}", update_count);
    eprintln!("");

    best_state
}

fn restore_steiner_paths(
    env: &Environment,
    state: &State,
    map: &MapState,
    safety_factor: f64,
) -> Vec<Vec<Coordinate>> {
    const INF: i32 = std::i32::MAX / 2;
    let edge_candidates = state.get_edge_candidates(env);
    let n = env.fixed_positions.len() + state.waypoints.len();
    let edges = mst::calculate_mst(&env.input, n, edge_candidates, MstEdgeAggregator::new());
    let mut paths = vec![];
    let positions = env
        .fixed_positions
        .iter()
        .chain(state.waypoints.iter())
        .copied()
        .collect_vec();

    for edge in edges.iter() {
        if edge.cost() == 0 {
            continue;
        }

        let start = edge.i();
        let goal = edge.j();
        let map_size = env.input.map_size;
        let mut distances = Map2d::new(vec![INF; map_size * map_size], map_size);
        let mut from = Map2d::new(vec![Coordinate::new(!0, !0); map_size * map_size], map_size);
        let mut queue = BinaryHeap::new();
        distances[positions[start]] = 0;
        queue.push(Reverse((0, positions[start])));

        while let Some(Reverse((dist, c))) = queue.pop() {
            if distances[c] < dist {
                continue;
            }

            for adj in ADJACENTS.iter() {
                let next = c + *adj;

                if next.in_map(map_size) {
                    let added_cost = if map.digged.is_digged(next) {
                        0
                    } else {
                        map.get_pred_cost(next, safety_factor, env.input)
                    };
                    let next_dist = dist + added_cost;

                    if distances[next].change_min(next_dist) {
                        queue.push(Reverse((next_dist, next)));
                        from[next] = c;
                    }
                }
            }
        }

        let mut c = positions[goal];
        let mut path = vec![c];

        while c != positions[start] {
            c = from[c];
            path.push(c);
        }

        paths.push(path);
    }

    paths
}
