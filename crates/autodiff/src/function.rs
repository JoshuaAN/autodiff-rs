use index_vec::IndexVec;

use crate::node::{Node, NodeId};

pub struct Function {
    nodes: IndexVec<NodeId, Node>,
    inputs: Vec<NodeId>,
    outputs: Vec<NodeId>,
}

impl Function {
    pub(crate) fn from_parts(
        nodes: IndexVec<NodeId, Node>,
        inputs: Vec<NodeId>,
        outputs: Vec<NodeId>,
    ) -> Self {
        Function {
            nodes,
            inputs,
            outputs,
        }
    }

    pub fn eval(&self, x: &[f64], out: &mut [f64]) {}

    pub fn nodes(&self) -> &IndexVec<NodeId, Node> {
        &self.nodes
    }
    pub fn inputs(&self) -> &[NodeId] {
        &self.inputs
    }
    pub fn outputs(&self) -> &[NodeId] {
        &self.outputs
    }
}
