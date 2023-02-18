use crate::common::grid::Coordinate;
use proconio::{input, source::line::LineSource};
use std::io::BufRead;

pub enum DiggingResult {
    /// 岩盤が破壊できていない
    NotBroken,
    /// 岩盤が破壊できた
    Broken(Coordinate),
    /// 全ての家に水が流れた
    Completed,
}

pub fn output(
    coordinate: Coordinate,
    power: i32,
    comment: &str,
    mut source: &mut LineSource<impl BufRead>,
) -> DiggingResult {
    println!("#{}", comment);
    println!("{} {} {}", coordinate.row, coordinate.col, power);

    input! {
        from &mut source,
        result: i32
    }

    match result {
        0 => DiggingResult::NotBroken,
        1 => DiggingResult::Broken(coordinate),
        2 => DiggingResult::Completed,
        _ => panic!("invalid output"),
    }
}
