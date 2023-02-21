use crate::{
    acl::dsu::Dsu,
    common::grid::{Coordinate, Map2d, ADJACENTS},
    input::Input,
    ChangeMinMax,
};
use itertools::izip;
use ndarray::{stack, Array3, Axis};

const SHRINK_RATIO: usize = 5;
const MINI_SIZE: usize = 40;
const MIN_STRENGTH: i32 = 10;
const MAX_STRENGTH: i32 = 5000;
const DEFAULT_STRENGTH: i32 = 100;

#[derive(Debug, Clone)]
pub struct MapState {
    pub digged: DiggedMap,
    pub damages: Map2d<i32>,
    pub predicted_strengths: Map2d<i32>,
    map_size: usize,
}

impl MapState {
    pub fn new(input: &Input) -> Self {
        let digged = DiggedMap::new(input.map_size, &input.waters);
        let damages = Map2d::new(vec![0; input.map_size * input.map_size], input.map_size);
        let predicted_strengths =
            Map2d::new(vec![DEFAULT_STRENGTH; MINI_SIZE * MINI_SIZE], MINI_SIZE);

        Self {
            digged,
            damages,
            predicted_strengths,
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

    pub fn export_tensor(&self) -> Array3<f32> {
        const INF: f32 = std::f32::MAX;
        const DIGGED: f32 = 1.0;
        const NOT_DIGGED: f32 = 0.0;

        // 40x40のマップについて、頑丈さ・破壊したかどうかを調べる
        let mut strength_shrinked = vec![INF; MINI_SIZE * MINI_SIZE];
        let mut digged_shrinked = vec![NOT_DIGGED; MINI_SIZE * MINI_SIZE];

        for row in 0..self.map_size {
            for col in 0..self.map_size {
                let c = Coordinate::new(row, col);
                if self.digged.is_digged(c) {
                    // 0～1に正規化
                    let val = self.damages[c] as f32 / MAX_STRENGTH as f32;
                    let c_shrink = (c / SHRINK_RATIO).to_index(MINI_SIZE);

                    // 頑丈さは同じ格子内のminを採用する
                    strength_shrinked[c_shrink].change_min(val);
                    digged_shrinked[c_shrink] = DIGGED;
                }
            }
        }

        // 平均値で埋める
        let mut digged_count = 0.0;
        let mut digged_strength = 0.0;

        for (&s, &dig) in izip!(&strength_shrinked, &digged_shrinked) {
            if dig == DIGGED {
                digged_count += 1.0;
                digged_strength += s;
            }
        }

        let average = if digged_count == 0.0 {
            DEFAULT_STRENGTH as f32
        } else {
            digged_strength / digged_count
        };

        for (s, &dig) in izip!(strength_shrinked.iter_mut(), &digged_shrinked) {
            if dig == NOT_DIGGED {
                *s = average;
            }
        }

        // 2チャンネルの画像にする
        let str_tensor =
            Array3::from_shape_vec([1, MINI_SIZE, MINI_SIZE], strength_shrinked).unwrap();
        let dig_tensor =
            Array3::from_shape_vec([1, MINI_SIZE, MINI_SIZE], digged_shrinked).unwrap();

        stack![Axis(0), str_tensor, dig_tensor]
    }

    pub fn import_tensor(&mut self, predicted: &Array3<f32>) {
        for row in 0..MINI_SIZE {
            for col in 0..MINI_SIZE {
                let c = Coordinate::new(row, col);
                let value = (predicted[[0, c.row, c.col]] * MAX_STRENGTH as f32).round() as i32;
                let value = value.clamp(MIN_STRENGTH, MAX_STRENGTH);
                self.predicted_strengths[c] = value;
            }
        }
    }

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

                if self.predicted_strengths[c] >= threshold {
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

    pub fn is_digged(&self, c: Coordinate) -> bool {
        (self.flags[c] & DIGGED_FLAG) > 0
    }

    pub fn is_connected(&mut self, c: Coordinate) -> bool {
        let c_index = c.to_index(self.map_size);
        self.dsu.same(c_index, self.water_master())
    }

    pub fn has_digged_nearby(&self, c: Coordinate, dist: usize) -> bool {
        let row0 = c.row.saturating_sub(dist);
        let row1 = (c.row + dist).min(self.map_size - 1);

        for row in row0..=row1 {
            // マンハッタン距離がdist以下の範囲を探したい
            let d = dist - (c.row as isize - row as isize).abs() as usize;
            let col0 = c.col.saturating_sub(d);
            let col1 = (c.col + d).min(self.map_size - 1);

            for col in col0..=col1 {
                if self.is_digged(Coordinate::new(row, col)) {
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
