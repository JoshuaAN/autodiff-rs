use index_vec::IndexVec;

use crate::{
    instruction::{Inst, InstructionData},
    value::Value,
};

pub struct Function {
    /// Program instructions.
    pub insts: IndexVec<Inst, InstructionData>,

    /// Program layout.
    pub layout: Vec<Inst>,

    /// Number of function arguments.
    pub num_params: usize,

    /// Function return values.
    pub returns: Vec<Value>,
}

impl Function {
    pub fn inst_data(&self, i: Inst) -> InstructionData {
        self.insts[i]
    }

    pub fn args(&self, i: Inst) -> Vec<Value> {
        match self.insts[i] {
            InstructionData::Constant(_) => vec![],
            InstructionData::Unary(_, a) => vec![a],
            InstructionData::Binary(_, a, b) => vec![a, b],
        }
    }

    pub fn layout(&self) -> &[Inst] {
        &self.layout
    }

    pub fn detached(&self) -> impl Iterator<Item = Inst> + '_ {
        let placed: std::collections::HashSet<Inst> = self.layout.iter().copied().collect();
        self.insts.indices().filter(move |i| !placed.contains(i))
    }
}
