use index_vec::IndexVec;

use crate::node::{Node, NodeId};

pub struct Function {
  nodes: IndexVec<NodeId, Node>,
  outputs: Vec<NodeId>,
}

impl Function {
  pub fn eval(&self, x: &[f64], out: &mut [f64]) {
    
  }
}