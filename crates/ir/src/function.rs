use index_vec::IndexVec;

use crate::{
    instruction::{Inst, InstructionData},
    value::{Value, ValueData},
};

pub struct Function {
    /// Program instructions.
    pub insts: IndexVec<Inst, InstructionData>,

    /// Maps values to their type and definition.
    pub values: IndexVec<Value, ValueData>,

    /// List of program instructions in order of their execution. This is the only thing
    /// touched by DCE to avoid modifying the arena-allocated list of instructions.
    pub layout: Vec<Inst>,

    /// Function arguments.
    pub params: Vec<Value>,

    /// Function return values.
    pub returns: Vec<Value>,
}

impl Function {
    pub fn inst_data(&self, i: Inst) -> InstructionData {
        self.insts[i]
    }

    pub fn args(&self) -> &[Value] {
        &self.params
    }

    pub fn args_mut(&mut self) -> &mut [Value] {
        &mut self.params
    }

    pub fn layout(&self) -> &[Inst] {
        &self.layout
    }

    pub fn detached(&self) -> impl Iterator<Item = Inst> + '_ {
        let placed: std::collections::HashSet<Inst> = self.layout.iter().copied().collect();
        self.insts.indices().filter(move |i| !placed.contains(i))
    }
}
