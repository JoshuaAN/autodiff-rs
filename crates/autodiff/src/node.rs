use index_vec::define_index_type;

use crate::{
    bits::Bits64,
    op::{BinaryOp, UnaryOp},
};

define_index_type! { pub struct NodeId = u32; }

#[derive(Clone, Copy)]
pub enum Node {
    /// Function parameter.
    Param(u32),

    /// Constant value.
    Constant(Bits64),

    /// Unary operation on the result of a node.
    Unary(UnaryOp, NodeId),

    /// Binary operation on the results of two nodes.
    Binary(BinaryOp, NodeId, NodeId),
}
