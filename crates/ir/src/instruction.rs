use index_vec::define_index_type;

use crate::{
    bits::Bits64,
    op::{BinaryOp, UnaryOp},
    value::Value,
};

define_index_type! { pub struct Inst = u32; }

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum InstructionData {
    Constant(Bits64),
    Unary(UnaryOp, Value),
    Binary(BinaryOp, Value, Value),
}
