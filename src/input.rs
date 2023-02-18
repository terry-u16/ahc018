use std::io::BufRead;

use proconio::{input, source::line::LineSource};

use crate::common::grid::Coordinate;

#[derive(Debug, Clone)]
pub struct Input {
    pub map_size: usize,
    pub water_count: usize,
    pub house_count: usize,
    pub exhausting_energy: i32,
    pub waters: Vec<Coordinate>,
    pub houses: Vec<Coordinate>,
}

impl Input {
    pub fn read(mut source: &mut LineSource<impl BufRead>) -> Self {
        input! {
            from &mut source,
            map_size: usize,
            water_count: usize,
            house_count: usize,
            exhausting_energy: i32,
        }

        let mut waters = vec![];

        for _ in 0..water_count {
            input! {
                from &mut source,
                row: usize,
                col: usize,
            }

            waters.push(Coordinate::new(row, col));
        }

        let mut houses = vec![];

        for _ in 0..water_count {
            input! {
                from &mut source,
                row: usize,
                col: usize,
            }

            houses.push(Coordinate::new(row, col));
        }

        Self {
            map_size,
            water_count,
            house_count,
            exhausting_energy,
            waters,
            houses,
        }
    }
}
