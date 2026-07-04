use crate::var::VarId;

use super::op::{BinaryOp, UnaryOp};

use index_vec::define_index_type;

define_index_type! { pub struct ExpressionId = u32; }

#[derive(Clone, Copy, Debug)]
pub enum Expression {
    Constant(f64),
    Variable(VarId),
    Unary(UnaryOp, ExpressionId),
    Binary(BinaryOp, ExpressionId, ExpressionId),
}
