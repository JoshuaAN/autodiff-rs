use crate::tape::{Tape, Var};

pub struct Gradient {}

pub struct Jacobian {}

pub struct Hessian {}

impl Gradient {
    pub fn new(tape: &Tape, inputs: &[Var], output: Var) -> Self {
        todo!("Implement gradient new")
    }

    pub fn eval(&self, x: &[f64], out: &mut [f64]) {
        todo!("Implement gradient evaluation")
    }
}

impl Jacobian {
    pub fn new(tape: &Tape, inputs: &[Var], outputs: &[Var]) -> Self {
        todo!("Implement Jacobian new")
    }

    pub fn eval(&self, x: &[f64]) {}
}

impl Hessian {
    pub fn new(tape: &Tape, inputs: &[Var], output: Var) -> Self {
        todo!("Implement Hessian new")
    }

    pub fn eval(&self, x: &[f64]) {}
}
