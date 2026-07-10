use index_vec::define_index_type;

use crate::{
    bits::Bits64,
    op::{BinaryOp, ReduceOp, UnaryOp},
    ty::{DimMap, DimSet, Ty},
    value::Value,
};

define_index_type! { pub struct Inst = u32; }

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum InstructionData {
    /// Represents a constant move: x <-- c.
    Constant(Bits64),

    /// Represents a unary expression, e.g. x <-- sin(a).
    Unary(UnaryOp, Value),

    /// Represents a binary expression, e.g. x <-- a + b.
    Binary(BinaryOp, Value, Value),

    /// Broadcasts the argument to the target type using Numpy broadcasting rules.
    Broadcast(Value, Ty, DimMap),

    /// Reduce over the given dimensions
    Reduce(ReduceOp, Value, DimSet),
}
