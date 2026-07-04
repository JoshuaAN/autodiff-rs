use index_vec::{IndexVec, define_index_type};

use crate::{
    op::{BinaryOp, UnaryOp},
    var::VarId,
};

define_index_type! { pub struct Slot = u32; }

#[derive(Clone, Copy, Debug)]
pub enum Instr {
    Const(f64),
    Input(VarId),
    Unary(UnaryOp, Slot),
    Binary(BinaryOp, Slot, Slot),
}

#[derive(Clone, Debug)]
pub struct Tape {
    pub insts: IndexVec<Slot, Instr>,
    pub outputs: Vec<Slot>,
}
