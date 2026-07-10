use std::collections::HashMap;

use index_vec::IndexVec;

use crate::{
    instruction::{Inst, InstructionData},
    ty::{Ty, TyData},
    value::{Value, ValueData},
};

#[derive(Clone, Default)]
pub struct Function {
    /// Program instructions.
    pub insts: IndexVec<Inst, InstructionData>,

    /// Stores the result of each instruction.
    pub results: IndexVec<Inst, Value>,

    /// Maps values to their type and definition.
    pub values: IndexVec<Value, ValueData>,

    /// Program types.
    types: IndexVec<Ty, TyData>,

    /// Maps type data to type indices. This helps avoid duplicate type data, so if two
    /// type indices are equal, then their data is also equal.
    type_map: HashMap<TyData, Ty>,

    /// List of program instructions in order of their execution. This is the only thing
    /// touched by DCE to avoid modifying the arena-allocated list of instructions.
    pub layout: Vec<Inst>,

    /// Function arguments.
    pub params: Vec<Value>,

    /// Function return values.
    pub returns: Vec<Value>,
}

impl Function {
    pub fn new(param_tys: &[TyData]) -> Self {
        let mut f = Function::default();
        for (num, &td) in param_tys.iter().enumerate() {
            assert!(num <= u16::MAX as usize, "too many params");
            let ty = f.intern_ty(td);
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

    pub fn inst_result(&self, i: Inst) -> Value {
        self.results[i]
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

    pub fn import_ty(&mut self, func: &Function, ty: Ty) -> Ty {
        self.intern_ty(func.ty_data(ty))
    }

    pub fn intern_ty(&mut self, td: TyData) -> Ty {
        *self
            .type_map
            .entry(td)
            .or_insert_with(|| self.types.push(td))
    }

    pub fn ty_data(&self, ty: Ty) -> TyData {
        self.types[ty]
    }

    pub fn value_ty(&self, v: Value) -> Ty {
        self.values[v].ty()
    }

    fn result_ty(&mut self, d: &InstructionData) -> Result<Ty, String> {
        Ok(match *d {
            InstructionData::Constant(_) => self.intern_ty(TyData::SCALAR),
            InstructionData::Unary(_, a) => self.value_ty(a),
            InstructionData::Binary(_, a, b) => {
                let (ta, tb) = (self.value_ty(a), self.value_ty(b));
                if ta != tb {
                    return Err(format!(
                        "binary op shape mismatch: {:?} vs {:?}",
                        self.ty_data(ta).dims(), self.ty_data(tb).dims()
                    ));
                }
                ta
            }
            InstructionData::Broadcast(v, ty) => {
                // let src = self.ty_data(self.value_ty(v));
                // let dst = self.ty_data(ty);
                // if map.dims().len() != src.rank() as usize {
                //     return Err("broadcast: map length != source rank".into());
                // }
                // let mut prev = None;
                // for (i, &o) in map.dims().iter().enumerate() {
                //     if prev.map_or(false, |p| o <= p) || o >= dst.rank() {
                //         return Err("broadcast: map must be strictly increasing, in range".into());
                //     }
                //     prev = Some(o);
                //     let (s, t) = (src.dims()[i], dst.dims()[o as usize]);
                //     if s != t && s != 1 {
                //         return Err(format!("broadcast: dim {i} incompatible ({s} vs {t})"));
                //     }
                // }
                // ty
                todo!("implement result_ty for broadcast")
            }
            InstructionData::Reduce(_, v, dims) => {
                todo!("implement result_ty for reduce")
            }
        })
    }


    pub fn emit(&mut self, d: InstructionData) -> Value {
        let ty = self.result_ty(&d).expect("emit: type error");
        let inst = self.insts.push(d);
        let v = self.values.push(ValueData::Result { ty, inst });
        let i2 = self.results.push(v);
        debug_assert_eq!(inst, i2);
        self.layout.push(inst);
        v
    }

    pub fn broadcast_to(&mut self, v: Value, target: Ty) -> Value {
        todo!("implement broadcast_to")
    }
}
