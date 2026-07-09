use index_vec::IndexVec;

use crate::{
    instruction::{Inst, InstructionData},
    ty::Ty,
    value::{Value, ValueData},
};

#[derive(Clone, Default)]
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
    pub fn new(param_tys: &[Ty]) -> Self {
        let mut f = Function::default();
        for (num, &ty) in param_tys.iter().enumerate() {
            let v = f.values.push(ValueData::Param {
                ty,
                num: num as u16,
            });
            f.params.push(v);
        }
        f
    }

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

    pub fn value_ty(&self, v: Value) -> Ty {
        self.values[v].ty()
    }

    fn result_ty(&self, d: &InstructionData) -> Result<Ty, String> {
        Ok(match *d {
            InstructionData::Constant(_) => Ty::SCALAR,
            InstructionData::Unary(_, a) => self.value_ty(a),
            InstructionData::Binary(_, a, b) => Ty::broadcast(self.value_ty(a), self.value_ty(b))?,
        })
    }

    pub fn emit(&mut self, d: InstructionData) -> Value {
        let ty = self.result_ty(&d).expect("emit: type error");
        let inst = self.insts.push(d);
        let v = self.values.push(ValueData::Result { ty, inst });
        self.layout.push(inst);
        v
    }
}
