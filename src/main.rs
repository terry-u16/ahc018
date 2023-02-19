mod common;
mod init_solution;
mod input;
mod network;
mod output;

use std::{
    io::{self, BufReader},
    time::Instant,
};

use init_solution::InitSolver;
use output::{output, DiggingResult};
use proconio::source::line::LineSource;
#[allow(unused_imports)]
use proconio::*;
#[allow(unused_imports)]
use rand::prelude::*;

use crate::input::Input;

pub trait ChangeMinMax {
    fn change_min(&mut self, v: Self) -> bool;
    fn change_max(&mut self, v: Self) -> bool;
}

impl<T: PartialOrd> ChangeMinMax for T {
    fn change_min(&mut self, v: T) -> bool {
        *self > v && {
            *self = v;
            true
        }
    }

    fn change_max(&mut self, v: T) -> bool {
        *self < v && {
            *self = v;
            true
        }
    }
}

#[allow(unused_macros)]
macro_rules! mat {
    ($e:expr; $d:expr) => { vec![$e; $d] };
    ($e:expr; $d:expr $(; $ds:expr)+) => { vec![mat![$e $(; $ds)*]; $d] };
}

fn main() {
    let mut stdin = LineSource::new(BufReader::new(io::stdin()));
    let input = Input::read(&mut stdin);
    let mut solver = InitSolver::new(&input);

    loop {
        let action = solver.get_next_action(&input);
        let result = output(action, "", &mut stdin);

        if let DiggingResult::Completed = result {
            break;
        }

        solver.update(result);
    }

    let elapsed = Instant::now() - input.since;
    eprintln!("{:.3}s", elapsed.as_secs_f64());
}
