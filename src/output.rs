use crate::common::grid::Coordinate;
use proconio::{input, source::line::LineSource};
use std::io::BufRead;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Action {
    pub coordinate: Coordinate,
    pub power: i32,
}

impl Action {
    pub fn new(coordinate: Coordinate, power: i32) -> Self {
        Self { coordinate, power }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiggingResult {
    /// 岩盤が破壊できていない
    NotBroken(Action),
    /// 岩盤が破壊できた
    Broken(Action),
    /// 全ての家に水が流れた
    Completed,
}

pub fn output(
    action: Action,
    comment: &str,
    mut source: &mut LineSource<impl BufRead>,
) -> DiggingResult {
    println!("#{}", comment);
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
