use crate::common::grid::Coordinate;
use proconio::{input, source::line::LineSource};
use std::io::BufRead;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Action {
    pub coordinate: Coordinate,
    pub power: i32,
    pub comment: Vec<String>,
}

impl Action {
    pub fn new(coordinate: Coordinate, power: i32, comment: Vec<String>) -> Self {
        Self {
            coordinate,
            power,
            comment,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiggingResult {
    /// 岩盤が破壊できていない
    NotBroken(Action),
    /// 岩盤が破壊できた
    Broken(Action),
    /// 全ての家に水が流れた
    Completed,
}

pub fn output(action: Action, mut source: &mut LineSource<impl BufRead>) -> DiggingResult {
    for s in action.comment.iter() {
        println!("#{}", &s);
    }
    println!(
        "{} {} {}",
        action.coordinate.row, action.coordinate.col, action.power
    );

    input! {
        from &mut source,
        result: i32
    }

    match result {
        0 => DiggingResult::NotBroken(action),
        1 => DiggingResult::Broken(action),
        2 => DiggingResult::Completed,
        _ => panic!("invalid output"),
    }
}
