use std::fmt;

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
    pub instrs: IndexVec<Slot, Instr>,
    pub outputs: Vec<Slot>,
}

impl fmt::Display for Tape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (slot, instr) in self.instrs.iter_enumerated() {
            match *instr {
                Instr::Const(c) => writeln!(f, "{:?} = const {}", slot, c)?,
                Instr::Input(v) => writeln!(f, "{:?} = input {:?}", slot, v)?,
                Instr::Unary(op, a) => writeln!(f, "{:?} = {:?} {:?}", slot, op, a)?,
                Instr::Binary(op, a, b) => writeln!(f, "{:?} = {:?} {:?} {:?}", slot, op, a, b)?,
            }
        }
        writeln!(f, "outputs: {:?}", self.outputs)
    }
}
