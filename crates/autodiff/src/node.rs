use crate::var::VarId;

use super::op::{BinaryOp, UnaryOp};

use index_vec::define_index_type;

define_index_type! { pub struct NodeId = u32; }

#[derive(Clone, Copy, Debug)]
pub enum Node {
    Constant(f64),
    Variable(VarId),
    Unary(UnaryOp, NodeId),
    Binary(BinaryOp, NodeId, NodeId),
}
