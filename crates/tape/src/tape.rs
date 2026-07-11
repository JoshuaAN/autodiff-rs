use index_vec::{IndexVec, define_index_type};

use crate::{
    bits::Bits64,
    op::{BinaryOp, UnaryOp},
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
}

define_index_type! { pub struct Value = u32; }

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ValueData {
    /// The value is the result of an instruction.
    Result(Inst),

    /// The value is a parameter of the function.
    Param(u16),
}

#[derive(Clone, Default)]
pub struct Tape {
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

impl Tape {
    pub fn inst_data(&self, i: Inst) -> InstructionData {
        self.insts[i]
    }

    pub fn args(&self) -> &[Value] {
        &self.params
    }

    pub fn args_mut(&mut self) -> &mut [Value] {
        &mut self.params
    }

    pub fn num_params(&self) -> usize {
        self.params.len()
    }

    pub fn layout(&self) -> &[Inst] {
        &self.layout
    }

    pub fn layout_mut(&mut self) -> &mut [Inst] {
        &mut self.layout
    }

    fn push(&mut self, data: InstructionData) -> Value {
        let inst = self.insts.push(data);
        let v = self.values.push(ValueData::Result(inst));
        self.layout.push(inst);
        v
    }

    pub fn param(&mut self) -> Value {
        let index = self.params.len() as u16;
        let v = self.values.push(ValueData::Param(index));
        self.params.push(v);
        v
    }

    pub fn constant(&mut self, x: f64) -> Value {
        self.push(InstructionData::Constant(Bits64::from_f64(x)))
    }

    pub fn unary(&mut self, op: UnaryOp, a: Value) -> Value {
        self.push(InstructionData::Unary(op, a))
    }

    pub fn binary(&mut self, op: BinaryOp, a: Value, b: Value) -> Value {
        self.push(InstructionData::Binary(op, a, b))
    }
}
