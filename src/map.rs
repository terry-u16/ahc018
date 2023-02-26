use crate::{
    acl::dsu::Dsu,
    common::grid::{Coordinate, Map2d, ADJACENTS},
    gaussean_process::GaussianPredictor,
    input::Input,
    ChangeMinMax,
};
use itertools::Itertools;
use nalgebra::{DMatrix, DVector};

const SHRINK_RATIO: usize = 5;
const MINI_SIZE: usize = 40;
const MAX_STRENGTH: i32 = 5000;

#[derive(Debug)]
pub struct MapState {
    pub digged: DiggedMap,
    pub damages: Map2d<i32>,
    prediction: PredictionResult,
    gaussean: GaussianPredictor,
    prediction_count: usize,
    map_size: usize,
}

impl MapState {
    pub fn new(input: &Input) -> Self {
        let digged = DiggedMap::new(input.map_size, &input.waters);
        let damages = Map2d::new(vec![0; input.map_size * input.map_size], input.map_size);

        Self {
            digged,
            damages,
            prediction: PredictionResult::null(),
            gaussean: GaussianPredictor::new(),
            prediction_count: 0,
            map_size: input.map_size,
        }
    }

    pub fn dig(&mut self, c: Coordinate, power: i32, broken: bool) {
        assert!(!self.digged.is_digged(c));

        self.damages[c] += power;
        self.damages[c].change_min(MAX_STRENGTH);

        if broken {
            self.digged.dig(c);
        }
    }

    pub fn mark_give_up(&mut self, c: Coordinate) {
        self.damages[c] = 2500;
        self.digged.mark_revealed(c);
    }

    pub fn get_pred_sturdiness(&self, c: Coordinate, sigma: f64) -> i32 {
        self.prediction.get_pred_value(c, sigma).round() as i32
    }

    pub fn get_pred_cost(&self, c: Coordinate, sigma: f64, input: &Input) -> i32 {
        self.get_pred_sturdiness(c, sigma) + input.exhausting_energy
    }

    pub fn update_prediction(&mut self, input: &Input) {
        self.set_data();
        let target_x = Self::get_target_points();

        let (mean, var) = self.gaussean.gaussian_process_regression(&target_x);

        self.prediction = PredictionResult::new(input, mean, var);

        self.prediction_count += 1;
    }

    fn set_data(&mut self) {
        self.gaussean.clear();

        for row in 0..self.map_size {
            for col in 0..self.map_size {
                let c = Coordinate::new(row, col);
                if self.digged.is_revealed(c) {
                    let x = DVector::from_vec(vec![row as f64, col as f64]);
                    let y = (self.damages[c] as f64).sqrt();
                    self.gaussean.add_data(x, y);
                }
            }
        }

        if self.prediction_count < 5 {
            // パラメータをグリッドサーチ
            let t1_cands = (3..10).map(|v| 2.0f64.powi(v)).collect_vec();
            let t2_cands = (2..12)
                .map(|v| {
                    let v = 5.0 * v as f64;
                    v * v
                })
                .collect_vec();
            let t3_cands = (0..6).map(|v| 2.0f64.powi(v)).collect_vec();
            self.gaussean
                .grid_search_theta(&t1_cands, &t2_cands, &t3_cands);
        }
    }

    fn get_target_points() -> DMatrix<f64> {
        let mut values = vec![];

        for row in 0..=MINI_SIZE {
            for col in 0..=MINI_SIZE {
                values.push((SHRINK_RATIO * row) as f64);
                values.push((SHRINK_RATIO * col) as f64);
            }
        }

        let len = (MINI_SIZE + 1) * (MINI_SIZE + 1);
        DMatrix::from_row_slice(len, 2, &values)
    }

    #[allow(dead_code)]
    pub fn dump_pred(&self, input: &Input, threshold: i32) {
        for row in 0..MINI_SIZE {
            'square: for col in 0..MINI_SIZE {
                let c = Coordinate::new(row, col);

                for &w in input.waters.iter() {
                    if c == w / 5 {
                        eprint!("[]");
                        continue 'square;
                    }
                }

                for &h in input.houses.iter() {
                    if c == h / 5 {
                        eprint!("<>");
                        continue 'square;
                    }
                }

                let c = Coordinate::new(row * 5, col * 5);

                if self.get_pred_sturdiness(c, 0.0) >= threshold {
                    eprint!("##");
                } else {
                    eprint!("  ");
                }
            }
            eprintln!();
        }
    }
}

const DIGGED_FLAG: u8 = 1 << 0;
const WATER_FLAG: u8 = 1 << 1;
const REVEALED_FLAG: u8 = 1 << 2;

#[derive(Debug, Clone)]
pub struct DiggedMap {
    flags: Map2d<u8>,
    dsu: Dsu,
    map_size: usize,
}

impl DiggedMap {
    pub fn new(map_size: usize, waters: &[Coordinate]) -> Self {
        // マスタ水源を超頂点と見なして管理する
        let dsu = Dsu::new(map_size * map_size + 1);
        let mut flags = Map2d::new(vec![0; map_size * map_size], map_size);
        for w in waters {
            flags[*w] |= WATER_FLAG;
        }

        Self {
            flags,
            dsu,
            map_size,
        }
    }

    pub fn dig(&mut self, c: Coordinate) {
        assert!((self.flags[c] & DIGGED_FLAG) == 0);
        self.flags[c] |= DIGGED_FLAG;
        self.mark_revealed(c);
        let c_index = c.to_index(self.map_size);

        // 水源だったら超頂点と繋ぐ
        if (self.flags[c] & WATER_FLAG) > 0 {
            self.dsu.merge(c_index, self.water_master());
        }

        for &adj in ADJACENTS.iter() {
            let next = c + adj;

            // 隣も掘られていたら繋ぐ
            if next.in_map(self.map_size) && (self.flags[next] & DIGGED_FLAG) > 0 {
                self.dsu.merge(c_index, next.to_index(self.map_size));
            }
        }
    }

    fn mark_revealed(&mut self, c: Coordinate) {
        self.flags[c] |= REVEALED_FLAG
    }

    pub fn is_digged(&self, c: Coordinate) -> bool {
        (self.flags[c] & DIGGED_FLAG) > 0
    }

    pub fn is_revealed(&self, c: Coordinate) -> bool {
        (self.flags[c] & REVEALED_FLAG) > 0
    }

    #[allow(dead_code)]
    pub fn is_connected(&mut self, c: Coordinate) -> bool {
        let c_index = c.to_index(self.map_size);
        self.dsu.same(c_index, self.water_master())
    }

    pub fn has_revealed_nearby(&self, c: Coordinate, dist: usize) -> bool {
        let row0 = c.row.saturating_sub(dist);
        let row1 = (c.row + dist).min(self.map_size - 1);

        for row in row0..=row1 {
            // マンハッタン距離がdist以下の範囲を探したい
            let d = dist - (c.row as isize - row as isize).abs() as usize;
            let col0 = c.col.saturating_sub(d);
            let col1 = (c.col + d).min(self.map_size - 1);

            for col in col0..=col1 {
                if self.is_revealed(Coordinate::new(row, col)) {
                    return true;
                }
            }
        }

        false
    }

    fn water_master(&self) -> usize {
        self.map_size * self.map_size
    }
}

#[derive(Debug, Clone)]
struct PredictionResult {
    mean_sqrt: Map2d<f64>,
    stddev_sqrt: Map2d<f64>,
}

impl PredictionResult {
    fn new(input: &Input, mean_sqrt: Vec<f64>, var_sqrt: Vec<f64>) -> Self {
        let (mean_sqrt, stddev_sqrt) = Self::convert(input, mean_sqrt, var_sqrt);
        Self {
            mean_sqrt,
            stddev_sqrt,
        }
    }

    fn null() -> Self {
        PredictionResult {
            mean_sqrt: Map2d::new(vec![], 1),
            stddev_sqrt: Map2d::new(vec![], 1),
        }
    }

    fn convert(
        input: &Input,
        mut mean_sqrt_raw: Vec<f64>,
        mut var_sqrt_raw: Vec<f64>,
    ) -> (Map2d<f64>, Map2d<f64>) {
        // sqrt(10)
        const LOWER: f64 = 3.16227766;

        for mean in mean_sqrt_raw.iter_mut() {
            mean.change_max(LOWER);
        }

        for var in var_sqrt_raw.iter_mut() {
            // TODO: たまに変な値になるので要調査
            var.change_max(1.0);
        }

        let mut stddev_sqrt = vec![];
        for var in var_sqrt_raw.iter() {
            stddev_sqrt.push(var.sqrt());
        }

        let mean_sqrt_raw = Map2d::new(mean_sqrt_raw, MINI_SIZE + 1);
        let stddev_sqrt_raw = Map2d::new(stddev_sqrt, MINI_SIZE + 1);

        let n = input.map_size;
        let mut mean_sqrt = Map2d::new(vec![0.0; n * n], n);
        let mut stddev_sqrt = Map2d::new(vec![0.0; n * n], n);

        for row in 0..n {
            for col in 0..n {
                let c = Coordinate::new(row, col);
                mean_sqrt[c] = Self::bilinear(&mean_sqrt_raw, c);
                stddev_sqrt[c] = Self::bilinear(&stddev_sqrt_raw, c);
            }
        }

        (mean_sqrt, stddev_sqrt)
    }

    fn bilinear(map: &Map2d<f64>, c: Coordinate) -> f64 {
        let r0 = c.row / SHRINK_RATIO;
        let r1 = r0 + 1;
        let c0 = c.col / SHRINK_RATIO;
        let c1 = c0 + 1;

        let v00 = map[Coordinate::new(r0, c0)];
        let v01 = map[Coordinate::new(r0, c1)];
        let v10 = map[Coordinate::new(r1, c0)];
        let v11 = map[Coordinate::new(r1, c1)];

        let dr = (c.row - r0 * SHRINK_RATIO) as f64 / SHRINK_RATIO as f64;
        let dc = (c.col - c0 * SHRINK_RATIO) as f64 / SHRINK_RATIO as f64;

        let v0 = v00 * (1.0 - dr) + v10 * dr;
        let v1 = v01 * (1.0 - dr) + v11 * dr;
        let v = v0 * (1.0 - dc) + v1 * dc;

        v
    }

    fn get_pred_value(&self, c: Coordinate, sigma: f64) -> f64 {
        let mean_sq = self.mean_sqrt[c];
        let stddev_sq = self.stddev_sqrt[c];
        let mut v = mean_sq + sigma * stddev_sq;

        // sqrt(10), sqrt(5000)
        const LOWER: f64 = 3.16227766;
        const UPPER: f64 = 70.71067811;
        v.change_max(LOWER);
        v.change_min(UPPER);

        // sqrtを取っていたので元に戻す
        v * v
    }
}
