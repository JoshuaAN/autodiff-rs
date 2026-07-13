use crate::tape::{Tape, Var};

/// Represents a sparse matrix in Compressed Sparse Column (CSC) format.
pub struct SparseStorage<T> {
  num_rows: usize,
  num_cols: usize,
  col_ptrs: Vec<u32>,
  row_idx: Vec<u32>,
  values: Vec<T>,
}

pub struct Gradient {

}

pub struct Jacobian {
  
}

pub struct Hessian {

}

impl Gradient {
  pub fn new(tape: &Tape, inputs: &[Var], output: &Var) -> Self {
    todo!("Implement gradient new")
  }

  pub fn eval(&self, x: &[f64]) -> Vec<f64> {
    todo!("Implement gradient evaluation")
  }
}

impl Jacobian {
  pub fn new(tape: &Tape, inputs: &[Var], outputs: &[Var]) -> Self {
    todo!("Implement Jacobian new")
  }

  pub fn eval(&self, x: &[f64]) -> SparseStorage<f64> {
    todo!("Implement Jacobian evaluation")
  }
}

impl Hessian {
  pub fn new(tape: &Tape, inputs: &[Var], output: &Var) -> Self {
    todo!("Implement Hessian new")
  }

  pub fn eval(&self, x: &[f64]) -> SparseStorage<f64> {
    todo!("Implement Hessian evaluation")
  }
}